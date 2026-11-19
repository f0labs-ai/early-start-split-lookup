//! delayline: a stand-in for DISTANCE between two services on one machine:
//! a TCP relay that holds every byte it forwards for a fixed time, each way.
//!
//!   delayline <listen-addr> <to-addr> [--delay-ms 20]
//!
//! A request and its answer take twice the delay longer than they would side
//! by side. Renting two machines in two places measures one distance, with
//! that day's route in it; a delay line measures any distance, the same one
//! every run, and nothing else changes. What it leaves out: a fresh
//! connection's handshake pays no delay (the callers it was used with keep
//! their connections open), and there is no loss and no jitter. It gives a
//! floor for what distance costs.
//!
//! Each connection gets two pumps, one per direction. A pump stamps what it
//! reads with the time it may leave, and a writer sends it then: the delay is
//! the same for every byte whatever the traffic, and order is kept.

use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::time::Instant;

#[derive(Parser)]
#[command(about = "A TCP relay that delays every byte by a fixed time each way.")]
struct Args {
    listen: String,
    to: String,
    /// The delay each way, in milliseconds.
    #[arg(long, env = "DELAY_MS", default_value_t = 20)]
    delay_ms: u64,
}

/// One direction: everything read from `from` reaches `to` `delay` later.
async fn pump(mut from: OwnedReadHalf, mut to: OwnedWriteHalf, delay: Duration) {
    let (tx, mut rx) = mpsc::unbounded_channel::<(Instant, Vec<u8>)>();
    let reader = tokio::spawn(async move {
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            match from.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if tx
                        .send((Instant::now() + delay, buf[..n].to_vec()))
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    });
    while let Some((due, bytes)) = rx.recv().await {
        // The delay being emulated, not a synchronisation: each chunk leaves
        // at its own due time.
        tokio::time::sleep_until(due).await;
        if to.write_all(&bytes).await.is_err() {
            break;
        }
    }
    // The reader ended and all it read has left: close this direction, as
    // late as its last byte.
    let _ = to.shutdown().await;
    reader.abort();
}

/// Relay one accepted connection to `to`, both ways.
async fn relay(client: TcpStream, to: String, delay: Duration) -> Result<()> {
    let upstream = TcpStream::connect(&to)
        .await
        .with_context(|| format!("connect {to}"))?;
    client.set_nodelay(true)?;
    upstream.set_nodelay(true)?;
    let (cr, cw) = client.into_split();
    let (ur, uw) = upstream.into_split();
    tokio::join!(pump(cr, uw, delay), pump(ur, cw, delay));
    Ok(())
}

/// Accept on `listener` forever, relaying each connection to `to`.
async fn serve(listener: TcpListener, to: String, delay: Duration) -> Result<()> {
    loop {
        let (client, _) = listener.accept().await?;
        let to = to.clone();
        tokio::spawn(async move {
            if let Err(e) = relay(client, to, delay).await {
                eprintln!("delayline: {e:#}");
            }
        });
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let args = Args::parse();
    let listener = TcpListener::bind(&args.listen)
        .await
        .with_context(|| format!("listen {}", args.listen))?;
    println!(
        "delayline on {} to {}, {} ms each way",
        args.listen, args.to, args.delay_ms
    );
    serve(listener, args.to, Duration::from_millis(args.delay_ms)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A server that sends back what it gets.
    async fn echo() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr").to_string();
        tokio::spawn(async move {
            while let Ok((mut s, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    while let Ok(n) = s.read(&mut buf).await {
                        if n == 0 || s.write_all(&buf[..n]).await.is_err() {
                            break;
                        }
                    }
                });
            }
        });
        addr
    }

    #[tokio::test]
    async fn every_byte_arrives_in_order_and_a_round_trip_pays_the_delay_twice() {
        let to = echo().await;
        let delay = Duration::from_millis(40);
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(serve(listener, to, delay));

        let mut c = TcpStream::connect(addr).await.expect("connect");
        for round in 0..3u8 {
            let sent: Vec<u8> = (0..200u8).map(|b| b.wrapping_add(round)).collect();
            let t = std::time::Instant::now();
            c.write_all(&sent).await.expect("write");
            let mut got = vec![0u8; sent.len()];
            c.read_exact(&mut got).await.expect("read");
            let took = t.elapsed();
            assert_eq!(got, sent, "round {round}: the same bytes, in order");
            assert!(
                took >= delay * 2,
                "round {round}: {took:?} is under two delays"
            );
            assert!(
                took < delay * 2 + Duration::from_millis(60),
                "round {round}: {took:?} is far over two delays"
            );
        }
    }
}
