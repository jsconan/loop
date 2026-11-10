#!/bin/sh
# Provision synthetic routed endpoints on a disposable Ubuntu VM for macOS tests.
set -eu

mode=${1:-}
root=/tmp/loop-macos-egress-fixture
address=11.0.0.2
http_unit=loop-macos-egress-http.service
https_unit=loop-macos-egress-https.service
dns_unit=loop-macos-egress-dns.service
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
server=$script_dir/../linux/egress_fixture.py
uid=$(id -u)

case "$mode" in
  setup)
    test "$(uname -s)" = Linux
    test "$uid" -ne 0
    test ! -e "$root"
    ! sudo -n ip -4 addr show dev lo | grep -q " $address/32 "
    test "$(systemctl is-active "$http_unit" || true)" != active
    test "$(systemctl is-active "$https_unit" || true)" != active
    mkdir -m 700 "$root"
    /usr/bin/python3 "$server" --prepare --directory "$root"
    openssl req -x509 -newkey rsa:2048 -nodes -days 2 -subj "/CN=$address" \
      -addext "subjectAltName=IP:$address" \
      -keyout "$root/key.pem" -out "$root/cert.pem" >/dev/null 2>&1
    sudo -n ip addr add "$address/32" dev lo
    sudo -n systemd-run --unit=loop-macos-egress-http --property="User=$(id -un)" \
      --property="WorkingDirectory=$root" /usr/bin/python3 "$server" \
      --address "$address" --port 18080 --directory "$root"
    sudo -n systemd-run --unit=loop-macos-egress-https --property="User=$(id -un)" \
      --property="WorkingDirectory=$root" /usr/bin/python3 "$server" \
      --address "$address" --port 18443 --directory "$root" \
      --certificate "$root/cert.pem" --key "$root/key.pem"
    systemctl is-active "$http_unit" "$https_unit"
    ;;
  cleanup)
    test "$(uname -s)" = Linux
    sudo -n systemctl stop "$http_unit" "$https_unit" "$dns_unit" || true
    sudo -n ip addr del "$address/32" dev lo || true
    if test -d "$root" && test "$(stat -c %u "$root")" = "$uid"; then
      rm -rf -- "$root"
    fi
    ;;
  dns-setup)
    : "${LOOP_MACOS_EGRESS_PRIVATE_IP:?Set the disposable VM private IPv4 address}"
    test -d "$root"
    test "$(stat -c %u "$root")" = "$uid"
    test "$(systemctl is-active "$dns_unit" || true)" != active
    : > "$root/dns-queries.log"
    sudo -n systemd-run --unit=loop-macos-egress-dns \
      --property="User=$(id -un)" \
      --property=AmbientCapabilities=CAP_NET_BIND_SERVICE \
      --property=CapabilityBoundingSet=CAP_NET_BIND_SERVICE \
      --property="WorkingDirectory=$root" /usr/bin/python3 \
      "$script_dir/egress_dns_fixture.py" \
      --address "$LOOP_MACOS_EGRESS_PRIVATE_IP" --directory "$root"
    systemctl is-active "$dns_unit"
    ;;
  dns-cleanup)
    sudo -n systemctl stop "$dns_unit" || true
    if test -d "$root" && test "$(stat -c %u "$root")" = "$uid"; then
      rm -f -- "$root/dns-queries.log"
    fi
    ;;
  *)
    printf 'Usage: %s setup|cleanup|dns-setup|dns-cleanup\n' "$0" >&2
    exit 2
    ;;
esac
