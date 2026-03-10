#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use jobs::connection::parse_redis_url;
use libfuzzer_sys::fuzz_target;
use url::Url;

#[derive(Arbitrary, Debug)]
struct RedisUrlSeed<'a> {
    payload: &'a [u8],
}

fuzz_target!(|data: &[u8]| {
    let Ok(seed) = RedisUrlSeed::arbitrary_take_rest(Unstructured::new(data)) else {
        return;
    };
    let input = String::from_utf8_lossy(seed.payload).into_owned();

    let result = parse_redis_url(&input);
    let result_again = parse_redis_url(&input);
    assert_eq!(result.is_ok(), result_again.is_ok());

    match (Url::parse(&input), result) {
        (Ok(parsed), Ok(config)) if matches!(parsed.scheme(), "redis" | "rediss") => {
            assert_eq!(config.tls, parsed.scheme() == "rediss");
            assert_eq!(config.host, parsed.host_str().unwrap_or("localhost"));
            assert_eq!(config.port, parsed.port().unwrap_or(6379));
            assert_eq!(
                config.database,
                parsed
                    .path()
                    .trim_start_matches('/')
                    .parse::<u32>()
                    .unwrap_or(0)
            );
            assert_eq!(
                config.username.as_deref(),
                (!parsed.username().is_empty()).then_some(parsed.username())
            );
            assert_eq!(config.password.as_deref(), parsed.password());
            assert!(!config.host.is_empty());
        }
        (Ok(parsed), Ok(_config)) => {
            panic!("accepted unsupported scheme: {}", parsed.scheme());
        }
        (Ok(parsed), Err(_)) => {
            assert!(!matches!(parsed.scheme(), "redis" | "rediss"));
        }
        (Err(_), Err(_)) => {}
        (Err(_), Ok(_)) => {
            panic!("accepted an invalid URL");
        }
    }
});
