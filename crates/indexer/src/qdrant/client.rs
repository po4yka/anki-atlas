use std::collections::HashMap;

use async_trait::async_trait;
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointId,
    PointStruct, RetrievedPoint, ScrollPointsBuilder, SearchPointsBuilder, Value,
    VectorParamsBuilder, vectors_config,
};
use sha2::{Digest, Sha256};

use super::repository::VectorRepository;
use super::schema::{
    NotePayload, ScoredNote, SearchFilters, SemanticSearchHit, SparseVector, VectorStoreError,
};

/// Concrete Qdrant implementation.
pub struct QdrantRepository {
    client: Qdrant,
    collection_name: String,
}

impl QdrantRepository {
    /// Connect to Qdrant and create repository.
    pub async fn new(url: &str, collection_name: &str) -> Result<Self, VectorStoreError> {
        let grpc_url = common::config::qdrant_grpc_url(url)
            .map_err(|error| VectorStoreError::Connection(error.to_string()))?;

        // Validate URL by parsing it
        reqwest::Url::parse(&grpc_url)
            .map_err(|e| VectorStoreError::Connection(format!("invalid URL: {e}")))?;

        // Try to connect to validate the URL is reachable
        let client = qdrant_client::Qdrant::from_url(&grpc_url)
            .build()
            .map_err(|e| VectorStoreError::Connection(e.to_string()))?;

        // Attempt a health check to verify connectivity
        client
            .health_check()
            .await
            .map_err(|e| VectorStoreError::Connection(e.to_string()))?;

        Ok(Self {
            client,
            collection_name: collection_name.to_string(),
        })
    }

    /// Wrap an existing Qdrant client.
    pub fn from_client(client: Qdrant, collection_name: impl Into<String>) -> Self {
        Self {
            client,
            collection_name: collection_name.into(),
        }
    }

    fn build_filters(&self, filters: &SearchFilters) -> Option<Filter> {
        let mut must = Vec::new();
        let mut must_not = Vec::new();

        if let Some(deck_names) = filters.deck_names.clone().filter(|items| !items.is_empty()) {
            must.push(Condition::matches("deck_names", deck_names));
        }
        if let Some(tags) = filters.tags.clone().filter(|items| !items.is_empty()) {
            must.push(Condition::matches("tags", tags));
        }
        if let Some(model_ids) = filters.model_ids.clone().filter(|items| !items.is_empty()) {
            must.push(Condition::matches("model_id", model_ids));
        }
        if filters.mature_only {
            must.push(Condition::matches("mature", true));
        }
        if let Some(min_reps) = filters.min_reps {
            must.push(Condition::range(
                "reps",
                qdrant_client::qdrant::Range {
                    gte: Some(f64::from(min_reps)),
                    ..Default::default()
                },
            ));
        }
        if let Some(max_lapses) = filters.max_lapses {
            must.push(Condition::range(
                "lapses",
                qdrant_client::qdrant::Range {
                    lte: Some(f64::from(max_lapses)),
                    ..Default::default()
                },
            ));
        }
        if let Some(deck_names_exclude) = filters
            .deck_names_exclude
            .clone()
            .filter(|items| !items.is_empty())
        {
            must_not.push(Condition::matches("deck_names", deck_names_exclude));
        }
        if let Some(tags_exclude) = filters
            .tags_exclude
            .clone()
            .filter(|items| !items.is_empty())
        {
            must_not.push(Condition::matches("tags", tags_exclude));
        }

        match (must.is_empty(), must_not.is_empty()) {
            (true, true) => None,
            (false, true) => Some(Filter::must(must)),
            (true, false) => Some(Filter::must_not(must_not)),
            (false, false) => Some(Filter {
                must,
                should: Vec::new(),
                must_not,
                min_should: None,
            }),
        }
    }

    fn note_id_filter(&self, note_ids: &[i64]) -> Option<Filter> {
        (!note_ids.is_empty())
            .then(|| Filter::must([Condition::matches("note_id", note_ids.to_vec())]))
    }

    fn point_id_from_chunk_id(&self, chunk_id: &str) -> PointId {
        let mut hasher = Sha256::new();
        hasher.update(chunk_id.as_bytes());
        let digest = hasher.finalize();
        let numeric_id = u64::from_be_bytes([
            digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        ]);
        numeric_id.into()
    }

    #[cfg(test)]
    fn note_id_from_point(&self, point_id: Option<PointId>) -> Result<i64, VectorStoreError> {
        match point_id.and_then(|id| id.point_id_options) {
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(value)) => {
                i64::try_from(value).map_err(|_| {
                    VectorStoreError::Client(format!("point id {value} does not fit into i64"))
                })
            }
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(value)) => {
                Err(VectorStoreError::Client(format!(
                    "uuid point ids are not supported for note-backed storage: {value}"
                )))
            }
            None => Err(VectorStoreError::Client(
                "Qdrant point id missing".to_string(),
            )),
        }
    }

    fn payload_from_map(
        &self,
        payload: HashMap<String, Value>,
    ) -> Result<NotePayload, VectorStoreError> {
        let payload = Payload::from(payload);
        payload
            .deserialize()
            .map_err(|error| VectorStoreError::Client(error.to_string()))
    }

    fn semantic_hit_from_payload(&self, payload: NotePayload, score: f32) -> SemanticSearchHit {
        SemanticSearchHit {
            note_id: payload.note_id,
            chunk_id: payload.chunk_id,
            chunk_kind: payload.chunk_kind,
            modality: payload.modality,
            source_field: payload.source_field,
            asset_rel_path: payload.asset_rel_path,
            mime_type: payload.mime_type,
            preview_label: payload.preview_label,
            score,
        }
    }

    async fn create_collection(&self, dimension: usize) -> Result<(), VectorStoreError> {
        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection_name)
                    .vectors_config(VectorParamsBuilder::new(dimension as u64, Distance::Cosine)),
            )
            .await
            .map_err(|error| VectorStoreError::Client(error.to_string()))?;
        Ok(())
    }

    async fn scroll_points(
        &self,
        filter: Filter,
        with_vectors: bool,
    ) -> Result<Vec<RetrievedPoint>, VectorStoreError> {
        let mut points = Vec::new();
        let mut offset: Option<PointId> = None;

        loop {
            let mut request = ScrollPointsBuilder::new(&self.collection_name)
                .filter(filter.clone())
                .limit(256)
                .with_payload(true)
                .with_vectors(with_vectors);

            if let Some(current_offset) = offset.clone() {
                request = request.offset(current_offset);
            }

            let response = self
                .client
                .scroll(request)
                .await
                .map_err(|error| VectorStoreError::Client(error.to_string()))?;
            let next_page_offset = response.next_page_offset.clone();
            points.extend(response.result);

            if next_page_offset.is_none() {
                break;
            }
            offset = next_page_offset;
        }

        Ok(points)
    }

    async fn find_text_primary_vector(
        &self,
        note_id: i64,
    ) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let filter = Filter::must([
            Condition::matches("note_id", note_id),
            Condition::matches("chunk_kind", "text_primary".to_string()),
        ]);
        let mut results = self.scroll_points(filter, true).await?;
        let Some(point) = results.pop() else {
            return Ok(None);
        };
        let Some(vectors) = point.vectors else {
            return Err(VectorStoreError::Client(
                "Qdrant point vectors missing".to_string(),
            ));
        };

        match vectors.get_vector() {
            Some(qdrant_client::qdrant::vector_output::Vector::Dense(dense)) => {
                Ok(Some(dense.data))
            }
            Some(other) => Err(VectorStoreError::Client(format!(
                "unsupported vector output for duplicate detection: {other:?}"
            ))),
            None => Err(VectorStoreError::Client(
                "Qdrant vectors output missing".to_string(),
            )),
        }
    }

    async fn collection_dimension_from_info(&self) -> Result<Option<usize>, VectorStoreError> {
        let exists = self
            .client
            .collection_exists(&self.collection_name)
            .await
            .map_err(|error| VectorStoreError::Connection(error.to_string()))?;
        if !exists {
            return Ok(None);
        }

        let info = self
            .client
            .collection_info(&self.collection_name)
            .await
            .map_err(|error| VectorStoreError::Client(error.to_string()))?;

        let dimension = info
            .result
            .and_then(|result| result.config)
            .and_then(|config| config.params)
            .and_then(|params| params.vectors_config)
            .and_then(|config| match config.config {
                Some(vectors_config::Config::Params(params)) => Some(params.size as usize),
                Some(vectors_config::Config::ParamsMap(map)) => map
                    .map
                    .into_values()
                    .next()
                    .map(|params| params.size as usize),
                None => None,
            });

        Ok(dimension)
    }

    /// Convert text into a hashed sparse vector (sha256 tokens, L2-normalized TF weights).
    pub fn text_to_sparse_vector(text: &str) -> SparseVector {
        // Tokenize: lowercase, keep only alphanumeric tokens
        let mut token_counts: HashMap<u32, f32> = HashMap::new();

        for token in text.split_whitespace() {
            let cleaned: String = token.chars().filter(|c| c.is_alphanumeric()).collect();
            let cleaned = cleaned.to_lowercase();
            if cleaned.is_empty() {
                continue;
            }

            // Hash token to u32 index using sha256
            let mut hasher = Sha256::new();
            hasher.update(cleaned.as_bytes());
            let hash = hasher.finalize();
            let index = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);

            *token_counts.entry(index).or_insert(0.0) += 1.0;
        }

        if token_counts.is_empty() {
            return SparseVector::default();
        }

        // Compute TF weights: 1.0 + ln(count)
        let mut pairs: Vec<(u32, f32)> = token_counts
            .into_iter()
            .map(|(idx, count)| (idx, 1.0 + count.ln()))
            .collect();

        // Sort by index
        pairs.sort_by_key(|(idx, _)| *idx);

        // L2 normalize
        let norm: f32 = pairs.iter().map(|(_, v)| v * v).sum::<f32>().sqrt();

        let (indices, values): (Vec<u32>, Vec<f32>) = if norm > 0.0 {
            pairs.into_iter().map(|(i, v)| (i, v / norm)).unzip()
        } else {
            pairs.into_iter().unzip()
        };

        SparseVector { indices, values }
    }
}

#[async_trait]
impl VectorRepository for QdrantRepository {
    async fn ensure_collection(&self, dimension: usize) -> Result<bool, VectorStoreError> {
        if let Some(existing_dimension) = self.collection_dimension_from_info().await? {
            if existing_dimension != dimension {
                return Err(VectorStoreError::DimensionMismatch {
                    collection: self.collection_name.clone(),
                    expected: existing_dimension,
                    actual: dimension,
                });
            }
            return Ok(false);
        }

        self.create_collection(dimension).await?;
        Ok(true)
    }

    async fn collection_dimension(&self) -> Result<Option<usize>, VectorStoreError> {
        self.collection_dimension_from_info().await
    }

    async fn recreate_collection(&self, dimension: usize) -> Result<(), VectorStoreError> {
        if self
            .client
            .collection_exists(&self.collection_name)
            .await
            .map_err(|error| VectorStoreError::Connection(error.to_string()))?
        {
            self.client
                .delete_collection(&self.collection_name)
                .await
                .map_err(|error| VectorStoreError::Client(error.to_string()))?;
        }
        self.create_collection(dimension).await
    }

    async fn upsert_vectors(
        &self,
        vectors: &[Vec<f32>],
        payloads: &[NotePayload],
        _sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, VectorStoreError> {
        if vectors.len() != payloads.len() {
            return Err(VectorStoreError::Client(
                "vectors and payloads must have the same length".to_string(),
            ));
        }
        let points = vectors
            .iter()
            .zip(payloads.iter())
            .map(|(vector, payload)| {
                let json = serde_json::to_value(payload)
                    .map_err(|error| VectorStoreError::Client(error.to_string()))?;
                let qdrant_payload = Payload::try_from(json)
                    .map_err(|error| VectorStoreError::Client(error.to_string()))?;
                Ok(PointStruct::new(
                    self.point_id_from_chunk_id(&payload.chunk_id),
                    vector.clone(),
                    qdrant_payload,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.client
            .upsert_points(
                qdrant_client::qdrant::UpsertPointsBuilder::new(&self.collection_name, points)
                    .wait(true),
            )
            .await
            .map_err(|error| VectorStoreError::Client(error.to_string()))?;
        Ok(vectors.len())
    }

    async fn delete_vectors(&self, note_ids: &[i64]) -> Result<usize, VectorStoreError> {
        if note_ids.is_empty() {
            return Ok(0);
        }
        let Some(filter) = self.note_id_filter(note_ids) else {
            return Ok(0);
        };
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .wait(true)
                    .points(filter),
            )
            .await
            .map_err(|error| VectorStoreError::Client(error.to_string()))?;
        Ok(note_ids.len())
    }

    async fn get_existing_hashes(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, String>, VectorStoreError> {
        if note_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let Some(filter) = self.note_id_filter(note_ids) else {
            return Ok(HashMap::new());
        };
        let response = self.scroll_points(filter, false).await?;
        let mut hashes = HashMap::new();
        for point in response {
            let payload = self.payload_from_map(point.payload)?;
            hashes
                .entry(payload.note_id)
                .or_insert(payload.content_hash);
        }
        Ok(hashes)
    }

    async fn search_chunks(
        &self,
        query_vector: &[f32],
        _query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SemanticSearchHit>, VectorStoreError> {
        let mut request =
            SearchPointsBuilder::new(&self.collection_name, query_vector.to_vec(), limit as u64)
                .with_payload(true)
                .with_vectors(false);
        if let Some(filter) = self.build_filters(filters) {
            request = request.filter(filter);
        }

        let response = self
            .client
            .search_points(request)
            .await
            .map_err(|error| VectorStoreError::Client(error.to_string()))?;

        response
            .result
            .into_iter()
            .map(|point| {
                let payload = self.payload_from_map(point.payload)?;
                Ok(self.semantic_hit_from_payload(payload, point.score))
            })
            .collect()
    }

    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<ScoredNote>, VectorStoreError> {
        let Some(query_vector) = self.find_text_primary_vector(note_id).await? else {
            return Ok(Vec::new());
        };

        let filters = SearchFilters {
            deck_names: deck_names.map(|items| items.to_vec()),
            tags: tags.map(|items| items.to_vec()),
            ..Default::default()
        };
        let similar_notes = self
            .search(
                &query_vector,
                None,
                limit.saturating_mul(4).max(limit),
                &filters,
            )
            .await?;

        Ok(similar_notes
            .into_iter()
            .filter(|hit| hit.note_id != note_id && hit.score >= min_score)
            .take(limit)
            .collect())
    }

    async fn close(&self) -> Result<(), VectorStoreError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::point_id;

    fn dummy_store() -> QdrantRepository {
        let client = Qdrant::from_url("http://localhost:0").build().unwrap();
        QdrantRepository::from_client(client, "test")
    }

    #[test]
    fn build_filters_all_none_returns_none() {
        let store = dummy_store();
        let filters = SearchFilters::default();
        assert!(store.build_filters(&filters).is_none());
    }

    #[test]
    fn build_filters_deck_names_only() {
        let store = dummy_store();
        let filters = SearchFilters {
            deck_names: Some(vec!["Deck1".into()]),
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert_eq!(filter.must.len(), 1);
        assert!(filter.must_not.is_empty());
    }

    #[test]
    fn build_filters_tags_exclude_only() {
        let store = dummy_store();
        let filters = SearchFilters {
            tags_exclude: Some(vec!["old".into()]),
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert!(filter.must.is_empty());
        assert_eq!(filter.must_not.len(), 1);
    }

    #[test]
    fn build_filters_mature_only_flag() {
        let store = dummy_store();
        let filters = SearchFilters {
            mature_only: true,
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert_eq!(filter.must.len(), 1);
    }

    #[test]
    fn build_filters_min_reps_range() {
        let store = dummy_store();
        let filters = SearchFilters {
            min_reps: Some(5),
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert_eq!(filter.must.len(), 1);
    }

    #[test]
    fn build_filters_max_lapses_range() {
        let store = dummy_store();
        let filters = SearchFilters {
            max_lapses: Some(3),
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert_eq!(filter.must.len(), 1);
    }

    #[test]
    fn build_filters_combined_must_and_must_not() {
        let store = dummy_store();
        let filters = SearchFilters {
            deck_names: Some(vec!["A".into()]),
            tags_exclude: Some(vec!["B".into()]),
            ..Default::default()
        };
        let filter = store.build_filters(&filters).unwrap();
        assert_eq!(filter.must.len(), 1);
        assert_eq!(filter.must_not.len(), 1);
    }

    #[test]
    fn build_filters_empty_vecs_treated_as_none() {
        let store = dummy_store();
        let filters = SearchFilters {
            deck_names: Some(vec![]),
            tags: Some(vec![]),
            tags_exclude: Some(vec![]),
            deck_names_exclude: Some(vec![]),
            model_ids: Some(vec![]),
            ..Default::default()
        };
        assert!(store.build_filters(&filters).is_none());
    }

    #[test]
    fn note_id_from_point_numeric_ok() {
        let store = dummy_store();
        let point = Some(qdrant_client::qdrant::PointId {
            point_id_options: Some(point_id::PointIdOptions::Num(42)),
        });
        assert_eq!(store.note_id_from_point(point).unwrap(), 42);
    }

    #[test]
    fn note_id_from_point_uuid_err() {
        let store = dummy_store();
        let point = Some(qdrant_client::qdrant::PointId {
            point_id_options: Some(point_id::PointIdOptions::Uuid(
                "550e8400-e29b-41d4-a716-446655440000".into(),
            )),
        });
        assert!(store.note_id_from_point(point).is_err());
    }

    #[test]
    fn note_id_from_point_none_err() {
        let store = dummy_store();
        assert!(store.note_id_from_point(None).is_err());
    }
}
