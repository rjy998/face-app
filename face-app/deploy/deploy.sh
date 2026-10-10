#!/usr/bin/env bash
# ============================================================
# 一键部署脚本（Ubuntu / Debian 云服务器）
#   用法：sudo bash deploy/deploy.sh your-domain.com
#   前置：域名已解析到本机 IP；国内服务器需先完成 ICP 备案
# ============================================================
set -euo pipefail

DOMAIN="${1:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

log() { printf '\033[1;32m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -n "$DOMAIN" ]] || die "用法：bash deploy/deploy.sh your-domain.com"
[[ $EUID -eq 0 ]] || die "请用 root 或 sudo 运行"

# ---------- 1. 安装 Docker ----------
if ! command -v docker >/dev/null 2>&1; then
  log "安装 Docker..."
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
else
  log "Docker 已安装：$(docker --version)"
fi

# ---------- 2. 准备环境变量 ----------
if [[ ! -f .env ]]; then
  log "生成 .env（请稍后填入 ARK_API_KEY 等密钥）"
  cp backend/.env.example .env
  JWT_SECRET="$(head -c 48 /dev/urandom | base64 | tr -d '=+/')"
  SIGNED_SECRET="$(head -c 48 /dev/urandom | base64 | tr -d '=+/')"
  MYSQL_ROOT_PASSWORD="$(head -c 24 /dev/urandom | base64 | tr -d '=+/')"
  MYSQL_PASSWORD="$(head -c 24 /dev/urandom | base64 | tr -d '=+/')"
  sed -i "s|^JWT_SECRET=.*|JWT_SECRET=${JWT_SECRET}|" .env
  sed -i "s|^SIGNED_URL_SECRET=.*|SIGNED_URL_SECRET=${SIGNED_SECRET}|" .env
  sed -i "s|^APP_ENV=.*|APP_ENV=prod|" .env
  sed -i "s|^DEBUG=.*|DEBUG=false|" .env
  {
    echo "MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD}"
    echo "MYSQL_PASSWORD=${MYSQL_PASSWORD}"
    echo "DOMAIN=${DOMAIN}"
  } >> .env
else
  log ".env 已存在，跳过生成"
fi

# 打开防火墙端口
if command -v ufw >/dev/null 2>&1; then
  ufw allow 80/tcp || true
  ufw allow 443/tcp || true
fi

# ---------- 3. 写入域名到 Nginx 配置 ----------
sed -i "s|server_name .*;|server_name ${DOMAIN};|g" deploy/nginx.conf
log "Nginx server_name 已设置为 ${DOMAIN}"

# ---------- 4. 首次签发证书（无证书时用自签临时证书让 Nginx 先起来）----------
mkdir -p docker/certs
if [[ ! -f docker/certs/fullchain.pem ]]; then
  log "生成临时自签证书（随后用 certbot 换成正式证书）"
  openssl req -x509 -nodes -newkey rsa:2048 -days 7 \
    -keyout docker/certs/privkey.pem -out docker/certs/fullchain.pem \
    -subj "/CN=${DOMAIN}" >/dev/null 2>&1
fi

# ---------- 5. 启动服务 ----------
log "构建并启动容器..."
docker compose -f docker/docker-compose.yml --env-file .env up -d --build

log "等待后端健康检查..."
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    log "后端已就绪"
    break
  fi
  sleep 3
  [[ $i -eq 30 ]] && warn "后端健康检查超时，请查看日志：docker compose -f docker/docker-compose.yml logs backend"
done

# ---------- 6. 申请正式证书 ----------
log "申请 Let's Encrypt 证书..."
docker run --rm \
  -v "$PROJECT_DIR/docker/certs:/etc/letsencrypt/live/${DOMAIN}" \
  -v "$PROJECT_DIR/docker/certbot-www:/var/www/certbot" \
  certbot/certbot certonly --webroot -w /var/www/certbot \
  -d "${DOMAIN}" --non-interactive --agree-tos --register-unsafely-without-email || \
  warn "证书申请失败（可能是域名未解析/未备案），不影响后端运行，可稍后重试"

docker compose -f docker/docker-compose.yml --env-file .env restart nginx || true

# ---------- 7. 结果 ----------
log "部署完成 ✅"
echo "  健康检查： curl -fsS https://${DOMAIN}/api/health"
echo "  接口文档： https://${DOMAIN}/docs （上线后建议关闭）"
echo "  查日志：   docker compose -f docker/docker-compose.yml logs -f backend"
echo "  证书续期： 建议加一条定时任务："
echo "    0 3 * * * docker run --rm -v $PROJECT_DIR/docker/certs:/etc/letsencrypt -v $PROJECT_DIR/docker/certbot-www:/var/www/certbot certbot/certbot renew --webroot -w /var/www/certbot && docker compose -f $PROJECT_DIR/docker/docker-compose.yml restart nginx"
echo
warn "别忘了：编辑 .env 填 ARK_API_KEY（豆包方舟密钥）后重启后端，解读才会由大模型生成"
