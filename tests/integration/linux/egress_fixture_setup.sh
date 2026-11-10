#!/bin/sh
# Own synthetic externally routed endpoints on a disposable Ubuntu test VM only.
set -eu

mode=${1:-}
root=/tmp/loop-egress-fixture
namespace=loop-egress-fixture
http_unit=loop-egress-fixture.service
https_unit=loop-egress-https-fixture.service
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
uid=$(id -u)
gid=$(id -g)

case "$mode" in
  setup)
    : "${LOOP_NATIVE_EGRESS_PRIVATE_IP:?Set the VM host-only private IPv4 address}"
    test "$uid" -ne 0
    test ! -e "$root"
    ! sudo -n ip netns list | grep -q "^$namespace "
    ! ip link show lgehost >/dev/null 2>&1
    test "$(systemctl is-active "$http_unit" || true)" != active
    test "$(systemctl is-active "$https_unit" || true)" != active
    mkdir -m 700 "$root"
    /usr/bin/python3 "$script_dir/egress_fixture.py" --prepare --directory "$root"
    openssl req -x509 -newkey rsa:2048 -nodes -days 2 -subj /CN=loop-egress-fixture \
      -keyout "$root/key.pem" -out "$root/cert.pem" >/dev/null 2>&1
    sudo -n ip netns add "$namespace"
    sudo -n ip link add lgehost type veth peer name lgepeer
    sudo -n ip link set lgepeer netns "$namespace"
    sudo -n ip addr add 11.0.0.1/32 dev lgehost
    sudo -n ip link set lgehost up
    sudo -n ip -n "$namespace" addr add 11.0.0.2/32 dev lgepeer
    sudo -n ip -n "$namespace" link set lo up
    sudo -n ip -n "$namespace" link set lgepeer up
    sudo -n ip route add 11.0.0.2/32 dev lgehost
    sudo -n ip -n "$namespace" route add 11.0.0.1/32 dev lgepeer
    printf '%s\n' \
      "$LOOP_NATIVE_EGRESS_PRIVATE_IP loop-egress-mixed.invalid # loop-egress-owned" \
      "11.0.0.2 loop-egress-mixed.invalid # loop-egress-owned" \
      "11.0.0.2 loop-egress-rebind.invalid # loop-egress-owned" \
      "$LOOP_NATIVE_EGRESS_PRIVATE_IP loop-egress-private.invalid # loop-egress-owned" | \
      sudo -n tee -a /etc/hosts >/dev/null
    sudo -n systemd-run --unit=loop-egress-fixture --property="WorkingDirectory=$root" \
      /usr/sbin/ip netns exec "$namespace" /usr/bin/setpriv --reuid="$uid" --regid="$gid" \
      --clear-groups /usr/bin/python3 "$script_dir/egress_fixture.py" \
      --address 11.0.0.2 --port 18080 --directory "$root"
    sudo -n systemd-run --unit=loop-egress-https-fixture --property="WorkingDirectory=$root" \
      /usr/sbin/ip netns exec "$namespace" /usr/bin/setpriv --reuid="$uid" --regid="$gid" \
      --clear-groups /usr/bin/python3 "$script_dir/egress_fixture.py" \
      --address 11.0.0.2 --port 18443 --directory "$root" \
      --certificate "$root/cert.pem" --key "$root/key.pem"
    ;;
  cleanup)
    sudo -n systemctl stop "$https_unit" "$http_unit" || true
    sudo -n ip link delete lgehost || true
    sudo -n ip netns delete "$namespace" || true
    sudo -n sed -i '/# loop-egress-owned$/d' /etc/hosts
    if test -d "$root" && test "$(stat -c %u "$root")" = "$uid"; then
      rm -rf -- "$root"
    fi
    ;;
  rebind-public|rebind-private)
    : "${LOOP_NATIVE_EGRESS_PRIVATE_IP:?Set the VM host-only private IPv4 address}"
    sudo -n ip netns list | grep -q "^$namespace "
    if test "$mode" = rebind-public; then
      address=11.0.0.2
    else
      address=$LOOP_NATIVE_EGRESS_PRIVATE_IP
    fi
    sudo -n sed -i '/ loop-egress-rebind.invalid # loop-egress-owned$/d' /etc/hosts
    printf '%s\n' "$address loop-egress-rebind.invalid # loop-egress-owned" | \
      sudo -n tee -a /etc/hosts >/dev/null
    ;;
  *)
    printf 'Usage: %s setup|cleanup|rebind-public|rebind-private\n' "$0" >&2
    exit 2
    ;;
esac
