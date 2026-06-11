import os
import json
import requests
from features import FEATURE_COLS

# ── Retention playbooks stored as RAG knowledge base ──────────────────────────
PLAYBOOKS = [
    {
        "id": "billing_failure",
        "trigger": "billing failures payment issue",
        "content": "When a customer has billing failures: offer a 1-click payment update link, waive the next month's fee if 2+ failures, and assign a CSM to follow up within 24 hours."
    },
    {
        "id": "low_engagement",
        "trigger": "inactive low logins low engagement",
        "content": "For disengaged customers: highlight 3 features they haven't used yet, invite them to a personalized onboarding call, and share a success story from a similar customer."
    },
    {
        "id": "support_overload",
        "trigger": "open support tickets unresolved issues",
        "content": "When support tickets are piling up: escalate to senior support, offer a dedicated support line, and acknowledge the frustration directly in outreach."
    },
    {
        "id": "plan_downgrade",
        "trigger": "plan downgrade cost concern pricing",
        "content": "For customers who downgraded: offer a 20% loyalty discount for staying on the Pro plan, show ROI data from similar companies, and offer a free 30-day trial of Enterprise features."
    },
    {
        "id": "low_nps",
        "trigger": "low nps unhappy dissatisfied",
        "content": "For low NPS customers: open with a genuine apology, ask one specific question about their pain point, offer a direct line to the VP of Customer Success."
    },
    {
        "id": "contract_ending",
        "trigger": "contract expiring renewal",
        "content": "When a contract is ending soon: offer early renewal at a locked-in rate, present a roadmap of upcoming features, and schedule a business review call."
    },
]

def build_rag_context(churn_drivers: list[str]) -> str:
    """Simple keyword-based retrieval from playbooks."""
    drivers_text = " ".join(churn_drivers).lower()
    matched = []
    for pb in PLAYBOOKS:
        if any(kw in drivers_text for kw in pb["trigger"].split()):
            matched.append(pb["content"])
    if not matched:
        matched = [PLAYBOOKS[1]["content"]]  # default: low engagement
    return "\n\n".join(matched[:2])  # top 2 relevant playbooks


def generate_retention_email(customer: dict, churn_score: float,
                              churn_drivers: list[str],
                              api_key: str = None) -> dict:
    """
    Generate a personalized retention email via the Anthropic API.
    Falls back to a template if no API key is provided.
    """
    rag_context = build_rag_context(churn_drivers)
    drivers_str = "; ".join(churn_drivers)

    prompt = f"""You are a Customer Success Manager writing a retention email.

CUSTOMER PROFILE:
- Name: {customer.get('name', 'Valued Customer')}
- Plan: {customer.get('plan', 'Pro')}
- Monthly spend: ${customer.get('monthly_spend', 79):.0f}
- Tenure: {customer.get('tenure_months', 12)} months
- Churn risk score: {churn_score:.0%}

TOP CHURN DRIVERS (from ML model):
{drivers_str}

RETENTION PLAYBOOK (internal guidance):
{rag_context}

Write a warm, concise retention email (120-150 words). Rules:
- Subject line first, then email body
- Personalize using their specific churn drivers
- Include ONE concrete offer (from the playbook)
- End with a soft CTA (reply, schedule a call, or click a link)
- Tone: human, empathetic, not salesy
- Format: Subject: <subject>\n\n<email body>
"""

    # Try Anthropic API if key is available
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 400,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=15,
            )
            if resp.status_code == 200:
                text = resp.json()["content"][0]["text"]
                lines = text.strip().split("\n", 1)
                subject = lines[0].replace("Subject:", "").strip()
                body    = lines[1].strip() if len(lines) > 1 else text
                return {"subject": subject, "body": body, "source": "claude"}
        except Exception as e:
            print(f"API call failed: {e}, using template fallback")

    # Template fallback (no API key needed)
    subject = f"We'd love to keep you, {customer.get('name','').split()[1] if ' ' in customer.get('name','') else 'there'}"
    body = f"""Hi {customer.get('name', 'there')},

I noticed your account hasn't been as active recently, and I wanted to personally reach out.

Based on your {customer.get('tenure_months', 0)}-month journey with us on the {customer.get('plan', 'Pro')} plan, I can see some friction points: {drivers_str}.

I'd love to help. Here's what I can offer:
→ A free 30-minute onboarding session to unlock features you haven't tried yet
→ A 20% loyalty discount applied immediately if you'd like to continue

Would you be open to a quick 15-minute call this week? Just reply to this email.

Best,
Alex from Customer Success"""

    return {"subject": subject, "body": body, "source": "template"}


if __name__ == "__main__":
    # Quick test
    test_customer = {
        "name": "Customer 42",
        "plan": "Pro",
        "monthly_spend": 79,
        "tenure_months": 8,
    }
    drivers = ["inactive for 21 days", "2 billing failures", "low NPS score of 3"]
    result = generate_retention_email(test_customer, 0.82, drivers)
    print(f"Subject: {result['subject']}\n")
    print(result['body'])
    print(f"\n[Source: {result['source']}]")
