"""Preserve the SF3 sample whose P1 mask is empty."""

RECORD_ID = "sf3_legacy_npc0:train:sf3_train_external_009645"
MASK_SUFFIX = "train/sources/sf3_train_external_002488_attempt0000/segments/3/masks/33608076.png"


def authorized_empty_mask(record, slot):
    return (
        record.data_format == "sf3_ready"
        and record.record_id == RECORD_ID
        and record.source_episode_id == "sf3_train_external_002488_attempt0000"
        and record.subjects[slot] == "P1"
        and record.engine_subject_ids[slot] == 33608076
        and record.x0_masks[slot].as_posix().endswith("/" + MASK_SUFFIX)
    )
