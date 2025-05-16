use std::{
    convert::Infallible,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use dashmap::{mapref::entry::Entry, DashMap};
use futures::FutureExt;
use tonic::{
    body::BoxBody,
    codegen::http::{HeaderMap, Request, Response}, // http-0.2 types
    server::NamedService,
};
use tower::Service;

use crate::{
    lighthouse::{Lighthouse, LighthouseOpt},
    torchftpb::lighthouse_service_server::LighthouseServiceServer,
};

/// Metadata header recognised by both client interceptor and this router.
pub const ROOM_ID_HEADER: &str = "room-id";

/// gRPC server for a single room (inner state = `Arc<Lighthouse>`).
type GrpcSvc = LighthouseServiceServer<Arc<Lighthouse>>;

#[derive(Clone)]
pub struct Router {
    rooms: Arc<DashMap<String, Arc<GrpcSvc>>>,
    tmpl_opt: LighthouseOpt,
}

impl Router {
    pub fn new(tmpl_opt: LighthouseOpt) -> Self {
        Self {
            rooms: Arc::new(DashMap::new()),
            tmpl_opt,
        }
    }

    fn room_id(hdrs: &HeaderMap) -> &str {
        hdrs.get(ROOM_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("default")
    }

    async fn room_service(
        rooms: Arc<DashMap<String, Arc<GrpcSvc>>>,
        tmpl: LighthouseOpt,
        id: &str,
    ) -> Arc<GrpcSvc> {
        if let Some(svc) = rooms.get(id) {
            return svc.clone();
        }

        // Build room state once.
        let lh = Lighthouse::new(tmpl.clone())
            .await
            .expect("failed to create Lighthouse");

        let svc_new = Arc::new(LighthouseServiceServer::new(lh));

        match rooms.entry(id.to_owned()) {
            Entry::Occupied(e) => e.get().clone(),
            Entry::Vacant(v) => {
                v.insert(svc_new.clone());
                svc_new
            }
        }
    }
}

// Tower::Service implementation
impl Service<Request<BoxBody>> for Router {
    type Response = Response<BoxBody>;
    type Error = Infallible;
    type Future =
        Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send + 'static>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Request<BoxBody>) -> Self::Future {
        let rooms = self.rooms.clone();
        let tmpl = self.tmpl_opt.clone();
        let room = Self::room_id(req.headers()).to_owned();

        async move {
            let svc_arc = Self::room_service(rooms, tmpl, &room).await;

            // `Arc<GrpcSvc>` itself isn’t a Service; clone the inner value.
            let mut svc = (*svc_arc).clone();
            let resp = svc
                .call(req)
                .await
                .map_err(|_e| -> Infallible { unreachable!() })?;

            Ok(resp)
        }
        .boxed()
    }
}

// Forward tonic’s NamedService marker
impl NamedService for Router {
    const NAME: &'static str = <GrpcSvc as NamedService>::NAME;
}
