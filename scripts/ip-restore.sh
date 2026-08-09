#!/bin/bash
source /etc/mlx-cluster.conf 2>/dev/null
ROLE="${NODE_ROLE:-0}"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [mlx-cluster] $1"; }
if [ "$ROLE" = "0" ]; then
    IP="192.168.0.1"; PEER="192.168.0.2"
    log "Primary: setting $IP on en3"
    ifconfig en3 inet "$IP" netmask 255.255.255.252
    route change "$PEER" -interface en3 2>/dev/null
else
    IP="192.168.0.2"; PEER="192.168.0.1"
    log "Secondary: bridge0 down, setting $IP on en3"
    ifconfig bridge0 down 2>/dev/null
    ifconfig en3 inet "$IP" netmask 255.255.255.252
    route change "$PEER" -interface en3 2>/dev/null
fi
sleep 2
ping -c1 -t3 "$PEER" >/dev/null 2>&1 && log "Peer $PEER reachable" || log "Peer $PEER not yet available"
