#!/usr/bin/env bash
set -euo pipefail

MIN_RAM=0
DISK=64

usage() {
    printf 'Usage: %s --min-ram <GB> [--disk <GB>]\n' "$0"
    printf '\nOptions:\n'
    printf '  --min-ram  Minimum RAM in GB (required)\n'
    printf '  --disk     Disk size in GB (default: 64)\n'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-ram)
            MIN_RAM="$2"
            shift 2
            ;;
        --disk)
            DISK="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ "$MIN_RAM" -eq 0 ]]; then
    echo "Error: --min-ram is required"
    usage
fi

echo "Searching for offers with at least ${MIN_RAM}GB RAM..."
OFFER_ID=$(vastai search offers "$MIN_RAM" | awk 'NR>1 {print $1}' | head -1)

if [[ -z "$OFFER_ID" ]]; then
    echo "No offers found with ${MIN_RAM}GB RAM"
    exit 1
fi

echo "Found offer: $OFFER_ID"
echo "Creating instance..."
vastai create instance "$OFFER_ID" \
    --image unsuspicious-industries/sas26-reproduction:latest \
    --disk "$DISK" \
    --ssh

echo "Done!"