#!/usr/bin/env bash
# Copy the Hardware team's arm calibrations (configs/calibration on the hardware branch) to
# $CALIB_DIR, and link the names LeRobot's dual-arm mode looks for. Run it once, and again
# whenever Hardware updates a calibration. It never calibrates an arm.
#
# Never recalibrate these arms, and never press ENTER on "use provided calibration file" for any
# arm except rakesh_leader (see configs/calibration/README.md on the hardware branch).
SKIP_CHECKS=1
source "$(dirname "$0")/lib.sh"
REPO="$(git -C "$VLA_DIR" rev-parse --show-toplevel)"

git -C "$REPO" fetch -q origin "${CALIBRATION_REF#origin/}" \
    || echo "could not fetch $CALIBRATION_REF (offline?); using the copy already fetched" >&2
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git -C "$REPO" archive "$CALIBRATION_REF" configs/calibration | tar -x -C "$tmp"
src="$tmp/configs/calibration"

mkdir -p "$FOLLOWER_CALIB" "$LEADER_CALIB"
for id in "$FOLLOWER_A_ID" "$FOLLOWER_B_ID"; do
    cp "$src/robots/so_follower/$id.json" "$FOLLOWER_CALIB/" || { echo "no calibration for $id on $CALIBRATION_REF" >&2; exit 1; }
done
for id in "$LEADER_A_ID" "$LEADER_B_ID"; do
    cp "$src/teleoperators/so_leader/$id.json" "$LEADER_CALIB/" || { echo "no calibration for $id on $CALIBRATION_REF" >&2; exit 1; }
done

# Dual mode: arm A is left, arm B is right
ln -sfn "$FOLLOWER_A_ID.json" "$FOLLOWER_CALIB/${BI_FOLLOWER_ID}_left.json"
ln -sfn "$FOLLOWER_B_ID.json" "$FOLLOWER_CALIB/${BI_FOLLOWER_ID}_right.json"
ln -sfn "$LEADER_A_ID.json"   "$LEADER_CALIB/${BI_LEADER_ID}_left.json"
ln -sfn "$LEADER_B_ID.json"   "$LEADER_CALIB/${BI_LEADER_ID}_right.json"

echo "calibrations from $CALIBRATION_REF ($(git -C "$REPO" rev-parse --short "$CALIBRATION_REF")) in $CALIB_DIR:"
find "$FOLLOWER_CALIB" "$LEADER_CALIB" -name '*.json' -printf '  %P%l\n' | sed 's/\.json\([a-z]\)/.json -> \1/' | sort
