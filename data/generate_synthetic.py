"""Generate the SYNTHETIC NEET-style dataset (seeded, reproducible).

Every college, rank and seat count below is invented. Nothing here is real
NEET/MCC/state counselling data. Run:  python data/generate_synthetic.py
"""
import csv
import random
from pathlib import Path

random.seed(42)
OUT = Path(__file__).parent

PREFIX = ["Meridian", "Kestrel", "Alder", "Brindle", "Corvane", "Dunmore", "Elmsworth", "Fenwick",
          "Garland", "Halcyon", "Ironwood", "Juniper", "Kingsley", "Larkspur", "Marlow", "Northgate",
          "Oakhurst", "Pinecrest", "Quillon", "Ravensdale", "Stonebridge", "Thornfield", "Umberlea",
          "Vantage", "Westmere", "Yarrowby", "Ashgrove", "Bexley", "Caldwell", "Delmont", "Everhart",
          "Foxmoor", "Glenrock", "Hartwell", "Islington", "Jasperton", "Kelvin", "Lockridge",
          "Montrose", "Nettlefield"]
SUFFIX = ["Medical College", "Institute of Medical Sciences", "Medical University"]
STATES = ["West Bengal", "Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Uttar Pradesh",
          "Rajasthan", "Gujarat", "Kerala", "Madhya Pradesh"]
CATS = ["General", "EWS", "OBC", "SC", "ST"]
CAT_MULT = {"General": 1.0, "EWS": 1.25, "OBC": 1.7, "SC": 7.0, "ST": 11.0}
YEARS = [2023, 2024, 2025]

types = ["Government"] * 18 + ["Private"] * 16 + ["Deemed"] * 6
random.shuffle(types)

colleges = []
for i, p in enumerate(PREFIX, 1):
    typ = types[i - 1]
    lo, hi = {"Government": (400, 30000), "Private": (25000, 180000), "Deemed": (20000, 220000)}[typ]
    colleges.append({
        "college_id": f"SYN-{i:03d}",
        "college": f"{p} {SUFFIX[i % 3]}",
        "state": STATES[i % len(STATES)],
        "type": typ,
        "courses": ["MBBS"] + (["BDS"] if i % 2 == 0 else []),
        "base": random.randint(lo, hi),
        "drift": {y: (1.0 if y == YEARS[0] else random.uniform(0.92, 1.08)) for y in YEARS},
    })

cut_rows, seat_rows = [], []
for c in colleges:
    quotas = ["All India"] if c["type"] == "Deemed" else ["All India", "State"]
    for course in c["courses"]:
        course_mult = 1.0 if course == "MBBS" else 2.5
        for q in quotas:
            q_mult = 1.0 if q == "All India" else 0.7
            for cat in CATS:
                level = c["base"] * course_mult * q_mult * CAT_MULT[cat]
                for y in YEARS:
                    f = 1.0
                    for yy in YEARS[1:]:  # drift compounds year over year
                        if yy <= y:
                            f *= c["drift"][yy]
                    rank = int(round(level * f * random.uniform(0.98, 1.02)))
                    cut_rows.append([c["college_id"], c["college"], c["state"], c["type"],
                                     course, q, cat, y, rank])
            total = random.choice([100, 150, 200, 250]) if course == "MBBS" else random.choice([50, 70, 100])
            if len(quotas) == 1:
                seats = total
            else:
                seats = round(total * 0.15) if q == "All India" else total - round(total * 0.15)
            seat_rows.append([c["college_id"], c["college"], c["state"], c["type"], course, q, 2025, seats])

with open(OUT / "cutoffs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["college_id", "college", "state", "type", "course", "quota", "category", "year", "closing_rank"])
    w.writerows(cut_rows)
with open(OUT / "seat_matrix.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["college_id", "college", "state", "type", "course", "quota", "year", "seats"])
    w.writerows(seat_rows)
print(f"colleges={len(colleges)} cutoff_rows={len(cut_rows)} seat_rows={len(seat_rows)}")
