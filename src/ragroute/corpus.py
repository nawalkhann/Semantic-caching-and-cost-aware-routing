"""Synthetic product-docs knowledge base and a labelled question workload.

The corpus describes a fictional SaaS product ("Nimbus") with five plans. Every
fact is generated from the PLANS table, so each benchmark question has an exact,
programmatically computed reference answer. Questions come in two difficulty
classes:

* easy  - single-fact lookups ("How much storage does Pro include?")
* hard  - comparisons, constraint solving and arithmetic over several facts

Each question intent has several paraphrase templates. Some easy questions are
deliberately verbose and some hard ones terse, so query length alone is not a
reliable difficulty signal for the router.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

PLANS: dict[str, dict] = {
    "Free": dict(price=0, storage=5, max_seats=3, support="community forum",
                 response_hours=None, rate_limit=60, history_days=7, sso=False,
                 audit_logs=False, uptime_sla=None, regions=1),
    "Starter": dict(price=8, storage=100, max_seats=10, support="email",
                    response_hours=48, rate_limit=300, history_days=30, sso=False,
                    audit_logs=False, uptime_sla=None, regions=1),
    "Pro": dict(price=15, storage=1000, max_seats=50, support="email and live chat",
                response_hours=24, rate_limit=1000, history_days=90, sso=False,
                audit_logs=True, uptime_sla=99.9, regions=2),
    "Business": dict(price=25, storage=3000, max_seats=500, support="24/7 phone",
                     response_hours=4, rate_limit=5000, history_days=180, sso=True,
                     audit_logs=True, uptime_sla=99.95, regions=4),
    "Enterprise": dict(price=45, storage=10000, max_seats=100000,
                       support="a dedicated success manager", response_hours=1,
                       rate_limit=20000, history_days=365, sso=True, audit_logs=True,
                       uptime_sla=99.99, regions=8),
}
PLAN_NAMES = list(PLANS)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Intent:
    """One distinct information need, with its paraphrases and ground truth."""

    intent_id: str
    difficulty: str  # "easy" | "hard"
    paraphrases: tuple[str, ...]
    answer: str
    facts: tuple[str, ...]  # tokens a correct answer must contain
    distractors: tuple[str, ...] = field(default=())  # plausible wrong answers


def _yn(b: bool) -> str:
    return "included" if b else "not available"


def build_documents() -> list[Document]:
    docs = []
    for name, p in PLANS.items():
        seats = "unlimited seats" if p["max_seats"] >= 100000 else f"up to {p['max_seats']} seats"
        resp = (f"with a first response within {p['response_hours']} hours"
                if p["response_hours"] else "with no guaranteed response time")
        sla = f"a {p['uptime_sla']}% uptime SLA" if p["uptime_sla"] else "no uptime SLA"
        text = (
            f"The Nimbus {name} plan costs ${p['price']} per seat per month and supports {seats}. "
            f"Each seat on {name} includes {p['storage']} GB of storage. "
            f"{name} customers get {p['support']} support {resp}. "
            f"The {name} API rate limit is {p['rate_limit']} requests per minute. "
            f"File version history on {name} is kept for {p['history_days']} days. "
            f"Single sign-on (SSO) is {_yn(p['sso'])} on {name}, and audit logs are {_yn(p['audit_logs'])}. "
            f"{name} comes with {sla} and data residency in {p['regions']} region"
            f"{'s' if p['regions'] > 1 else ''}."
        )
        docs.append(Document(f"plan-{name.lower()}", f"{name} plan", text))
    docs += [
        Document("policy-refunds", "Refund policy",
                 "Annual subscriptions can be refunded in full within 30 days of purchase. "
                 "Monthly subscriptions can be refunded within 14 days. Refunds are issued to "
                 "the original payment method within 5 business days."),
        Document("policy-security", "Security overview",
                 "All Nimbus data is encrypted at rest with AES-256 and in transit with TLS 1.3. "
                 "Nimbus is SOC 2 Type II certified and runs annual third-party penetration tests."),
        Document("policy-export", "Data export",
                 "Workspace owners can export all files and metadata as a ZIP archive at any time. "
                 "Exports larger than 50 GB are delivered as multiple parts."),
        Document("policy-billing", "Billing",
                 "Annual billing gives a 20% discount compared to monthly billing. Seats can be "
                 "added at any time and are prorated to the end of the billing period."),
    ]
    return docs


# --------------------------------------------------------------------------- intents

_EASY_ATTRS = {
    # attr: (paraphrase templates, answer fn, fact fn)
    "price": (
        ["How much does the {p} plan cost per user?",
         "What's the price of {p} per seat each month?",
         "{p} plan monthly price per user?",
         "I'm putting together a budget for next year and need to know what Nimbus charges "
         "per seat per month on the {p} tier.",
         "how much is {p} per seat"],
        lambda p: f"The {p} plan costs ${PLANS[p]['price']} per seat per month.",
        lambda p: (str(PLANS[p]["price"]),),
    ),
    "storage": (
        ["How much storage does the {p} plan include?",
         "What is the storage limit per seat on {p}?",
         "{p} storage per user?",
         "Our designers keep a lot of large files around, so before we sign up I'd like to "
         "confirm how many gigabytes each seat gets on the {p} plan.",
         "how many GB do you get on {p}"],
        lambda p: f"Each seat on {p} includes {PLANS[p]['storage']} GB of storage.",
        lambda p: (str(PLANS[p]["storage"]),),
    ),
    "rate_limit": (
        ["What is the API rate limit on the {p} plan?",
         "How many API requests per minute does {p} allow?",
         "{p} API rate limit?",
         "We're building an integration that polls Nimbus frequently; what's the maximum "
         "number of API calls per minute on {p}?"],
        lambda p: f"The {p} API rate limit is {PLANS[p]['rate_limit']} requests per minute.",
        lambda p: (str(PLANS[p]["rate_limit"]),),
    ),
    "history": (
        ["How long is version history kept on {p}?",
         "How many days of file version history does the {p} plan keep?",
         "{p} version history retention?",
         "If someone overwrites a document by mistake, how far back can we restore old "
         "versions on the {p} plan?"],
        lambda p: f"File version history on {p} is kept for {PLANS[p]['history_days']} days.",
        lambda p: (str(PLANS[p]["history_days"]),),
    ),
    "sso": (
        ["Does the {p} plan include SSO?",
         "Is single sign-on available on {p}?",
         "{p} SSO support?",
         "Our IT team requires everyone to log in through our identity provider; can we set "
         "up single sign-on on the {p} plan?"],
        lambda p: f"SSO is {_yn(PLANS[p]['sso'])} on {p}.",
        lambda p: ((_yn(PLANS[p]["sso"]),)),
    ),
    "support": (
        ["What kind of support does the {p} plan get?",
         "What support channel comes with {p}?",
         "{p} support level?",
         "When something breaks for us, how do we reach Nimbus support on {p} and how fast "
         "do they reply?"],
        lambda p: (f"{p} includes {PLANS[p]['support']} support"
                   + (f" with a {PLANS[p]['response_hours']} hour response time."
                      if PLANS[p]["response_hours"] else ".")),
        lambda p: ((str(PLANS[p]["response_hours"]),) if PLANS[p]["response_hours"]
                   else ("community",)),
    ),
    "seats": (
        ["How many seats can the {p} plan have?",
         "What is the maximum number of users on {p}?",
         "{p} seat limit?",
         "We expect the team to keep growing over the next couple of years, so what's the "
         "most users we could have on the {p} plan?"],
        lambda p: ("{p} supports unlimited seats." if PLANS[p]["max_seats"] >= 100000
                   else f"{p} supports up to {PLANS[p]['max_seats']} seats.").format(p=p),
        lambda p: (("unlimited",) if PLANS[p]["max_seats"] >= 100000
                   else (str(PLANS[p]["max_seats"]),)),
    ),
    "uptime": (
        ["What uptime SLA does the {p} plan have?",
         "Is there an uptime guarantee on {p}?",
         "{p} uptime SLA?",
         "Our customers depend on files being reachable, so what availability does Nimbus "
         "commit to contractually on the {p} plan?"],
        lambda p: (f"{p} comes with a {PLANS[p]['uptime_sla']}% uptime SLA."
                   if PLANS[p]["uptime_sla"] else f"{p} has no uptime SLA."),
        lambda p: ((str(PLANS[p]["uptime_sla"]),) if PLANS[p]["uptime_sla"] else ("no",)),
    ),
}

_POLICY_INTENTS = [
    ("refund-annual", ["What is the refund window for annual subscriptions?",
                       "Can I get a refund on an annual plan, and for how long?",
                       "annual plan refund period?"],
     "Annual subscriptions can be refunded in full within 30 days.", ("30",)),
    ("refund-monthly", ["How long do I have to get a refund on a monthly subscription?",
                        "What's the refund window for monthly billing?",
                        "monthly subscription refund window?"],
     "Monthly subscriptions can be refunded within 14 days.", ("14",)),
    ("encryption", ["How is data encrypted at rest?",
                    "What encryption does Nimbus use for stored data?",
                    "encryption at rest?"],
     "Data is encrypted at rest with AES-256.", ("256",)),
    ("annual-discount", ["How much do I save with annual billing?",
                         "What discount do you get for paying yearly?",
                         "annual billing discount?"],
     "Annual billing gives a 20% discount.", ("20",)),
]

_COMPARE_ATTRS = {
    "storage": ("storage per seat", "GB", "storage"),
    "rate_limit": ("API rate limit", "requests per minute", "rate_limit"),
    "history": ("version history", "days", "history_days"),
    "price": ("price per seat", "dollars", "price"),
}

_COMPARE_TEMPLATES = [
    "Which has more {label}, {a} or {b}, and by how much?",
    "Compare the {label} of the {a} and {b} plans. What's the difference?",
    "{a} vs {b} {label}?",
    "How does {a} stack up against {b} on {label}, and what's the gap?",
]

_CONSTRAINT_TEMPLATES = [
    "My team has {n} people and we need {feature}. What's the cheapest plan that works and "
    "what would it cost per month?",
    "Cheapest plan for {n} users with {feature}, and monthly total?",
    "We are {n} people and {feature} is a must. Which plan should we pick to minimise cost, "
    "and how much is that each month?",
    "If we need {feature} for {n} seats, what is the least expensive option and its monthly bill?",
]

_ANNUAL_TEMPLATES = [
    "What would {n} seats on {p} cost per year with annual billing?",
    "Annual cost for {n} {p} seats paid yearly?",
    "If we buy {n} seats of {p} on the annual plan, what is the yearly total after the discount?",
    "{n} users on {p}, billed annually - total per year?",
]


def _cheapest(n: int, feature: str) -> str:
    key = {"SSO": "sso", "audit logs": "audit_logs", "an uptime SLA": "uptime_sla"}[feature]
    for name in PLAN_NAMES:  # ordered by price
        p = PLANS[name]
        if p["max_seats"] >= n and p[key]:
            return name
    raise ValueError("no plan")


def build_intents() -> list[Intent]:
    intents: list[Intent] = []
    for attr, (templates, ans, facts) in _EASY_ATTRS.items():
        for p in PLAN_NAMES:
            others = [q for q in PLAN_NAMES if q != p]
            intents.append(Intent(
                f"easy-{attr}-{p.lower()}", "easy",
                tuple(t.format(p=p) for t in templates), ans(p), facts(p) + (p,),
                tuple(ans(q) for q in others),
            ))
    for iid, templates, answer, facts in _POLICY_INTENTS:
        intents.append(Intent(f"easy-{iid}", "easy", tuple(templates), answer, facts,
                              ("Refunds are not offered.", "Please contact sales.")))

    for attr, (label, unit, key) in _COMPARE_ATTRS.items():
        for a, b in zip(PLAN_NAMES, PLAN_NAMES[1:]):
            va, vb = PLANS[a][key], PLANS[b][key]
            win, lose = (a, b) if va > vb else (b, a)
            diff = abs(va - vb)
            answer = f"{win} has more {label} than {lose}, by {diff} {unit}."
            wrong = f"{lose} has more {label} than {win}, by {diff} {unit}."
            intents.append(Intent(
                f"hard-compare-{attr}-{a.lower()}-{b.lower()}", "hard",
                tuple(t.format(label=label, a=a, b=b) for t in _COMPARE_TEMPLATES),
                answer, (f"{win} has more", str(diff)), (wrong, f"{win} has more {label}, by {diff * 2} {unit}."),
            ))

    for n, feature in [(5, "SSO"), (40, "SSO"), (8, "audit logs"), (60, "audit logs"),
                       (20, "an uptime SLA"), (600, "SSO"), (2, "audit logs"),
                       (300, "an uptime SLA")]:
        plan = _cheapest(n, feature)
        total = n * PLANS[plan]["price"]
        i = PLAN_NAMES.index(plan)
        alt = PLAN_NAMES[i + 1 if i + 1 < len(PLAN_NAMES) else i - 1]
        intents.append(Intent(
            f"hard-cheapest-{feature.split()[-1].lower()}-{n}", "hard",
            tuple(t.format(n=n, feature=feature) for t in _CONSTRAINT_TEMPLATES),
            f"{plan} is the cheapest plan with {feature} for {n} seats, at ${total} per month.",
            (plan, str(total)),
            (f"{alt} is the cheapest option, at ${n * PLANS[alt]['price']} per month.",),
        ))

    for p, n in [("Starter", 6), ("Pro", 12), ("Pro", 30), ("Business", 75), ("Enterprise", 200)]:
        total = round(n * PLANS[p]["price"] * 12 * 0.8)
        intents.append(Intent(
            f"hard-annual-{p.lower()}-{n}", "hard",
            tuple(t.format(n=n, p=p) for t in _ANNUAL_TEMPLATES),
            f"{n} seats of {p} cost ${total} per year with the 20% annual discount.",
            (str(total),),
            (f"{n} seats of {p} cost ${n * PLANS[p]['price'] * 12} per year.",),
        ))
    return intents


@dataclass(frozen=True)
class WorkloadItem:
    query: str
    intent: Intent


_PREFIXES = ["", "", "", "", "Quick question: ", "hi, ", "Hey team - ", "Hello! ",
             "Sorry if this was asked before, but ", "Can you tell me: ", "One more thing: "]
_SUFFIXES = ["", "", "", "", " Thanks!", " thx", " Thanks in advance.", " (asking for our team)",
             " Need this for a proposal."]
_SWAPS = [("plan", "tier"), ("seat", "user"), ("How much", "how much"), ("What's", "What is"),
          ("cost", "price"), ("include", "come with")]


def _noise(q: str, rng: random.Random) -> str:
    """Surface variation real users add: casing, punctuation, filler, small word swaps.
    None of it changes the meaning or the entities in the question."""
    for a, b in _SWAPS:
        if a in q and rng.random() < 0.25:
            q = q.replace(a, b, 1)
    if rng.random() < 0.2:
        q = q.rstrip("?")
    prefix = rng.choice(_PREFIXES)
    if prefix:
        q = prefix + q[0].lower() + q[1:]
    q += rng.choice(_SUFFIXES)
    if rng.random() < 0.15:
        q = q.lower()
    return q


def build_workload(n: int = 3000, seed: int = 7, zipf_s: float = 0.9) -> list[WorkloadItem]:
    """Sample a request stream where popular intents repeat (Zipf-distributed),
    each time with a random paraphrase and surface noise."""
    rng = random.Random(seed)
    intents = build_intents()
    order = intents[:]
    rng.shuffle(order)  # popularity rank independent of difficulty
    weights = [1.0 / (rank + 1) ** zipf_s for rank in range(len(order))]
    items = []
    for intent in rng.choices(order, weights=weights, k=n):
        items.append(WorkloadItem(_noise(rng.choice(intent.paraphrases), rng), intent))
    return items
