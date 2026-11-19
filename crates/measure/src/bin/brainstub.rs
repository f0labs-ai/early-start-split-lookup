//! brainstub: a stand-in language model for timing runs. An OpenAI-style
//! `/chat/completions` that streams a reply as server-sent events
//! (`data: {"choices":[{"delta":{"content":...}}]}` ... `data: [DONE]`) after
//! a fixed time to the first token.
//!
//!   brainstub [--listen 127.0.0.1:8098] [--ttft-ms 300] [--token-ms 15]
//!             [--reply "Here is answer number {n} for you."]
//!
//! Why: a timing run must measure the voice stack, not a hosted model's
//! queue. With this as the brain, the brain's hop is a known constant, and
//! two arms differ only in what is being measured. `{n}` in the reply becomes
//! the request's number, so no two replies are the same (an agent that
//! refuses to say a whole reply twice would otherwise ask again).
//!
//! Plain tokio: one request per connection, the response closes it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Result, bail};
use clap::Parser;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Parser)]
#[command(about = "A stand-in language model with a fixed time to the first token.")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8098")]
    listen: String,
    #[arg(long, env = "STUB_TTFT_MS", default_value_t = 300)]
    ttft_ms: u64,
    /// Between tokens: 15 ms is about 66 tokens a second.
    #[arg(long, env = "STUB_TOKEN_MS", default_value_t = 15)]
    token_ms: u64,
    #[arg(
        long,
        env = "STUB_REPLY",
        default_value = "Here is answer number {n} for you."
    )]
    reply: String,
}

/// The reply for request number `n`: its `{n}` filled in.
fn numbered(reply: &str, n: usize) -> String {
    reply.replace("{n}", &n.to_string())
}

/// Read one HTTP request (headers and a Content-Length body) and discard it.
async fn read_request(s: &mut TcpStream) -> Result<()> {
    let mut buf = Vec::with_capacity(16 * 1024);
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        let n = s.read(&mut chunk).await?;
        if n == 0 {
            bail!("the client closed before the headers ended");
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        if buf.len() > 1 << 20 {
            bail!("headers over 1 MiB");
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_ascii_lowercase();
    let want = head
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let mut have = buf.len() - header_end;
    while have < want {
        let n = s.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        have += n;
    }
    Ok(())
}

async fn serve(mut s: TcpStream, reply: String, ttft: Duration, per_token: Duration) -> Result<()> {
    read_request(&mut s).await?;
    // The emulated model's latency, not a synchronisation.
    tokio::time::sleep(ttft).await;
    s.write_all(
        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\nconnection: close\r\n\r\n",
    )
    .await?;
    for (i, word) in reply.split_inclusive(' ').enumerate() {
        if i > 0 {
            tokio::time::sleep(per_token).await;
        }
        let frame = serde_json::json!({"choices": [{"delta": {"content": word}}]});
        s.write_all(format!("data: {frame}\n\n").as_bytes()).await?;
    }
    s.write_all(b"data: [DONE]\n\n").await?;
    s.shutdown().await?;
    Ok(())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let args = Args::parse();
    let ttft = Duration::from_millis(args.ttft_ms);
    let per_token = Duration::from_millis(args.token_ms);
    let listener = TcpListener::bind(&args.listen).await?;
    println!(
        "brainstub on {}, first token after {ttft:?}, {per_token:?} a token",
        args.listen
    );
    let served = Arc::new(AtomicUsize::new(0));
    let reply: Arc<str> = Arc::from(args.reply.as_str());
    loop {
        let (s, _) = listener.accept().await?;
        let n = served.fetch_add(1, Ordering::Relaxed) + 1;
        let text = numbered(&reply, n);
        tokio::spawn(async move {
            if let Err(e) = serve(s, text, ttft, per_token).await {
                eprintln!("brainstub: {e:#}");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_numbered_reply_differs_for_every_request_and_a_plain_one_never() {
        assert_eq!(
            numbered("Here is answer number {n} for you.", 7),
            "Here is answer number 7 for you."
        );
        assert_ne!(numbered("Answer {n}.", 1), numbered("Answer {n}.", 2));
        assert_eq!(numbered("Sure.", 3), "Sure.");
    }

    #[tokio::test]
    async fn the_reply_streams_after_the_first_token_time() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let (s, _) = listener.accept().await.expect("accept");
            serve(
                s,
                "one two".into(),
                Duration::from_millis(50),
                Duration::from_millis(5),
            )
            .await
            .expect("serve");
        });
        let mut c = TcpStream::connect(addr).await.expect("connect");
        let t = std::time::Instant::now();
        c.write_all(b"POST /v1/chat/completions HTTP/1.1\r\ncontent-length: 2\r\n\r\n{}")
            .await
            .expect("write");
        let mut body = String::new();
        c.read_to_string(&mut body).await.expect("read");
        assert!(t.elapsed() >= Duration::from_millis(50));
        assert!(body.contains("\"content\":\"one \"") && body.contains("\"content\":\"two\""));
        assert!(body.trim_end().ends_with("data: [DONE]"));
    }
}
