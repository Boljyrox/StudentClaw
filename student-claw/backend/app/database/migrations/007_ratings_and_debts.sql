-- 007: food ratings, materialised debts, and /splitexpense support.
--
--  * bills.kind distinguishes an OCR'd receipt from a hand-entered expense, so
--    /splitbill and /splitexpense can share one split + settle engine.
--  * bill_ratings holds each member's 1-5 star vote on the food.
--  * bill_debts freezes who-owes-whom at finalise time and tracks payment,
--    which is what the cross-bill "pending expenses" summary aggregates.

ALTER TABLE bills
    ADD COLUMN IF NOT EXISTS kind VARCHAR(16) NOT NULL DEFAULT 'receipt';

CREATE TABLE IF NOT EXISTS bill_ratings (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bill_id    UUID NOT NULL REFERENCES bills(id) ON DELETE CASCADE,
    user_id    BIGINT NOT NULL,
    user_name  VARCHAR(100) NOT NULL,
    stars      SMALLINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_bill_rating_user UNIQUE (bill_id, user_id)
);

CREATE INDEX IF NOT EXISTS ix_bill_ratings_bill_id ON bill_ratings (bill_id);

CREATE TABLE IF NOT EXISTS bill_debts (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bill_id          UUID NOT NULL REFERENCES bills(id) ON DELETE CASCADE,
    chat_id          BIGINT NOT NULL,
    debtor_user_id   BIGINT,
    debtor_name      VARCHAR(100) NOT NULL,
    creditor_user_id BIGINT,
    creditor_name    VARCHAR(100) NOT NULL,
    amount           NUMERIC(10, 2) NOT NULL,
    paid_at          TIMESTAMPTZ,
    confirmed_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_bill_debts_chat_id ON bill_debts (chat_id);
CREATE INDEX IF NOT EXISTS ix_bill_debts_bill_id ON bill_debts (bill_id);
