# ProtocolNerd — AWS Deployment

The live system runs as a single container on **ECS Fargate** behind an ALB, with
**Cloudflare** in front. There is no EC2 instance and nothing to SSH into: a deploy
is build → push → new task-definition revision → service update.

```
Cloudflare (protocolnerd.wirelessnerd.org, TLS)
  └─> ALB ecs-express-gateway-alb  (HTTP :80 target group)
        └─> ECS Fargate service protocolsnerd  (container :8080)
              image: <account>.dkr.ecr.us-east-1.amazonaws.com/protocolsnerd:<git sha>
```

| Piece | Value |
|---|---|
| Region | `us-east-1` |
| ECR repository | `protocolsnerd` (images tagged with the short git SHA) |
| ECS cluster / service | `protocolsnerd` / `protocolsnerd` |
| Task-definition family | `protocolsnerd-protocolsnerd` |
| Task size | 1 vCPU, 2 GB (Fargate, `awsvpc`) |
| Container port | 8080 (`uvicorn main:app`, `PORT` env respected) |
| Frontend | served by the same container; Cloudflare fronts the ALB |

The Docker build **bakes the search index into the image**: `scripts/build_index.py`
and `scripts/build_dense_index.py` run at build time over the committed corpus in
`data/`, so the container boots with retrieval ready and has no runtime dependency
on an index store. This is also why builds are slow (tens of minutes on an Apple
Silicon machine, which emulates linux/amd64) and why a corpus refresh is a
redeploy (see below).

## Prerequisites

- AWS CLI configured for account `<ACCOUNT_ID>` with ECR/ECS permissions
- Docker
- A clean checkout of the commit you intend to ship (the image is tagged with it)

## Deploying

From the repository root:

```bash
REPO=<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/protocolsnerd
SHA=$(git rev-parse --short HEAD)

# 1. Build. The build args stamp the image so /health and the debug banner can
#    report exactly what is running; without them the banner shows "unknown".
docker build --platform linux/amd64 \
  --build-arg BUILD_SHA=$SHA \
  --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -t $REPO:$SHA .

# 2. Push.
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin ${REPO%%/*}
docker push $REPO:$SHA

# 3. New task-definition revision: current one with the image swapped.
aws ecs describe-task-definition --region us-east-1 \
  --task-definition protocolsnerd-protocolsnerd \
  --query taskDefinition > /tmp/td.json
python3 - <<EOF
import json
td = json.load(open("/tmp/td.json"))
for k in ("taskDefinitionArn","revision","status","requiresAttributes",
          "compatibilities","registeredAt","registeredBy"):
    td.pop(k, None)
td["containerDefinitions"][0]["image"] = "$REPO:$SHA"
json.dump(td, open("/tmp/td.json","w"))
EOF
aws ecs register-task-definition --region us-east-1 --cli-input-json file:///tmp/td.json \
  --query taskDefinition.revision --output text

# 4. Point the service at it (use the revision printed above).
aws ecs update-service --region us-east-1 --cluster protocolsnerd \
  --service protocolsnerd --task-definition protocolsnerd-protocolsnerd:<REV>
```

The rollout takes 10–12 minutes including connection draining on the old task.

## Verifying

```bash
# The stamp must match what you shipped; "source": "image" means it was baked in.
curl -s https://protocolnerd.wirelessnerd.org/health | python3 -m json.tool

# Per-domain source pairing (biology: protocols.io + PubMed;
# chemistry: protocols.io + Europe PMC). Check the "source" field of results.
curl -s -X POST https://protocolnerd.wirelessnerd.org/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Suzuki coupling of aryl boronic acid","search_mode":"local",
       "top_k":3,"explain":false,"no_log":true,
       "skip_clarification":true,"search_confirmed":true}'
```

## Configuration

Environment (task definition, plain values):

| Variable | Purpose |
|---|---|
| `ENABLE_EUROPEPMC` | Kill switch for the Europe PMC lane. `1` in production; `0` forces it off everywhere. Chemistry declares the source; biology never sees it either way. |
| `QUERY_LOGGING_ENABLED`, `QUERY_LOG_S3_BUCKET`, `QUERY_LOG_S3_PREFIX`, `QUERY_LOG_VIEWER_OPEN` | Query logging to S3 and the log viewer. |

Secrets (task definition `secrets`, from Secrets Manager): `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `PROTOCOLS_IO_TOKEN`.

Model choice, ranking flags, and the rest of the runtime configuration follow the
code defaults, which match the shipped configuration the paper evaluates (see
`render.yaml` for the same settings spelled out explicitly).

## Rolling back

Every deploy is a task-definition revision, so rollback is pointing the service at
the previous one:

```bash
aws ecs update-service --region us-east-1 --cluster protocolsnerd \
  --service protocolsnerd --task-definition protocolsnerd-protocolsnerd:<PREVIOUS_REV>
```

## Refreshing the corpus

The corpus lives in the repository (`data/protocols/`, one JSON per protocol) and
the index is baked at image build time, so a refresh is: re-run the crawl
(`scripts/fetch_protocols.py` / `scripts/retrieve_all_protocols.sh`), commit the
updated `data/`, and deploy as above. **There is no scheduled update in the
current architecture** — nothing refreshes the corpus until someone does this.


## Custom domain: protocolnerd.org

Cloudflare (registrar + DNS, proxied, Full strict) serves the domain through a
Worker (`deploy/cloudflare-worker.js`, routes `protocolnerd.org/*` and
`www.protocolnerd.org/*`) that proxies to the ECS Express service hostname.
That hostname rides the routing rule ECS rewrites on every blue/green deploy,
so the domain follows each deployment automatically and needs no post-deploy
action. Verified by deliberately breaking the fallback path with the Worker
live: the site stayed up.

As dormant fallback, apex and `www` CNAME records point at the ALB's DNS name,
an ACM certificate for both names is attached to the HTTPS listener via SNI,
and listener rule priority 10 routes the two hostnames to the service's target
groups. The fallback only matters if the Worker routes are removed, and then
this caveat applies:

**Post-deploy step, required only when serving via the fallback rule:** ECS Express flips traffic between a blue/green
pair of target groups on every deployment by rewriting its own rule's weights.
The protocolnerd.org rule does not get rewritten, so after every rollout copy
the ForwardConfig weights from the express rule (priority 1) to the domain rule
(priority 10):

```bash
L=$(aws elbv2 describe-listeners --region us-east-1 \
  --load-balancer-arn <ALB_ARN> --query "Listeners[0].ListenerArn" --output text)
CFG=$(aws elbv2 describe-rules --region us-east-1 --listener-arn "$L" \
  --query "Rules[?Priority=='1'].Actions[0]" --output json | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)))")
aws elbv2 modify-rule --region us-east-1 \
  --rule-arn $(aws elbv2 describe-rules --region us-east-1 --listener-arn "$L" --query "Rules[?Priority=='10'].RuleArn" --output text) \
  --actions "$CFG"
```

With the Worker in place this sync is unnecessary; it is kept for the day the
fallback path is in use.

## Legacy files in this directory

`backend_startup.sh`, `incremental_update.py`, `protocols-update.service`, and
`protocols-update.timer` belong to an earlier EC2-based deployment (a git clone
on an instance, systemd-managed uvicorn, and a weekly corpus-update timer). That
infrastructure no longer exists; the files are kept for reference and none of
them play any role in the Fargate deployment described here.
