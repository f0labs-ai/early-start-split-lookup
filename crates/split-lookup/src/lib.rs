//! A reference split lookup, built on open components.
//!
//! A lookup for a retrieval-grounded voice agent embeds the question, runs a
//! hybrid search (a dense leg and a keyword leg, fused by reciprocal rank),
//! reranks the candidates with a cross-encoder and keeps the top few that
//! clear a score gate. When the vector database sits apart from the box that
//! runs the models, the split puts every step that needs a model on that box
//! and asks the database ONE question per lookup:
//!
//! ```text
//!   the box with the models                 the vector database (no models)
//!   ------------------------------------    -----------------------------------
//!   1. embed the question
//!   2. shortlist from the passages'
//!      one-bit codes, held in memory
//!   3. --- one request: the dense --------> hybrid search: the dense leg
//!          vector, the keyword vector,      confined to the shortlist and
//!          the shortlisted ids              scored exactly; the keyword leg;
//!                                           reciprocal-rank fusion
//!      <-- the candidates, with text ------
//!   4. rerank and gate
//! ```
//!
//! [`Mode::Graph`] runs the same lookup with the database's own approximate
//! dense search (its graph over quantized vectors, rescored), so the two can
//! be compared answer for answer. A split lookup that cannot shortlist (no
//! codes, a vector of the wrong size) or whose request the database refuses
//! runs as [`Mode::Graph`] and says so: it is never worse than the lookup it
//! replaces.
//!
//! * [`bm25`]: the keyword vectors, query and passage side.
//! * [`contract`]: the boundary. [`Embed`], [`Rerank`] and [`Store`] are the
//!   three services a lookup depends on; any implementation will do.
//! * [`tei`]: the embedder and reranker over the HTTP interface of Hugging
//!   Face's text-embeddings-inference, and an embedder that replays saved
//!   query vectors.
//! * [`qdrant`]: the store, over Qdrant's REST interface.
//! * [`lookup`]: the lookup itself.

pub mod bm25;
pub mod contract;
pub mod lookup;
pub mod qdrant;
pub mod tei;

pub use contract::{
    Candidate, DenseLeg, Embed, Hit, HybridQuery, Rerank, ServiceError, SparseVec, Store,
};
pub use lookup::{Lookup, LookupConfig, LookupError, Mode, SearchOutcome, SearchRequest, Timings};
