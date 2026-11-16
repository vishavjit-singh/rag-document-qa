"""Content of the SYNTHETIC corpus: nine policy and product documents of a fictional company.

"Kestrelway Logistics" does not exist. Every name, number and rule below was written for this demo.
Blocks that carry a fact used by the gold QA set are tagged with a fact key; the PDF builder records the
page each tagged block lands on, so the gold set's page numbers come from the layout engine, not from
the extraction code being evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

COMPANY = "Kestrelway Logistics"
SYNTHETIC_NOTE = ("SYNTHETIC DOCUMENT. Kestrelway Logistics is a fictional company; this file was generated "
                  "for a portfolio demo and describes no real organisation.")
CONFIDENTIALITY = ("This document is intended for Kestrelway Logistics staff and approved partners. Do not forward "
                   "it outside the company without written approval from the document owner. Printed copies are "
                   "uncontrolled; the version on the policy portal is the one that applies.")


@dataclass
class Block:
    kind: str                       # "p" | "bullets" | "table"
    text: str = ""
    items: list[str] = field(default_factory=list)
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    fact: str | None = None         # fact key for a paragraph / bullet list / whole table
    row_facts: dict[int, str] = field(default_factory=dict)


@dataclass
class Section:
    heading: str
    blocks: list[Block]


@dataclass
class Chapter:
    title: str
    sections: list[Section]


@dataclass
class DocSpec:
    file: str
    code: str
    title: str
    version: str
    effective: str
    status: str                     # "Current" | "Superseded"
    owner: str
    chapters: list[Chapter]
    supersedes: str = "None"
    superseded_by: str = ""


@dataclass
class QA:
    id: str
    question: str
    fact: str | None                # None = unanswerable from the corpus
    answer: str = ""                # expected answer span (must appear in the fact's block)


def P(text, fact=None):
    return Block("p", text=text, fact=fact)


def B(items, fact=None):
    return Block("bullets", items=items, fact=fact)


def T(header, rows, fact=None, row_facts=None):
    return Block("table", header=header, rows=rows, fact=fact, row_facts=row_facts or {})


def S(heading, *blocks):
    return Section(heading, list(blocks))


def intro(purpose: str, scope: str) -> Chapter:
    return Chapter("Introduction", [
        S("Purpose", P(purpose)),
        S("Scope", P(scope)),
        S("Confidentiality notice", P(CONFIDENTIALITY)),
    ])


# --------------------------------------------------------------------------------------------- HR handbook
HR = DocSpec(
    file="employee-handbook-2025.pdf", code="KL-HR-001", title="Employee Handbook", version="2025.1",
    effective="1 April 2025", status="Current", owner="People Operations",
    chapters=[
        intro("This handbook explains the working arrangements, leave entitlements and people processes that apply "
              "to employees of Kestrelway Logistics.",
              "It applies to all permanent and fixed-term employees in India. Contractors and agency staff are "
              "covered by their own agreements and are not in scope of this handbook."),
        Chapter("Working hours and attendance", [
            S("Standard working hours",
              P("The standard working week is 40 hours, spread over five days from Monday to Friday.", "hr.week"),
              P("Core hours are 10:00 to 16:00 IST. Outside core hours, employees may start and finish at times "
                "agreed with their manager.", "hr.core"),
              P("Warehouse and transport roles work on published shift rosters instead of core hours.")),
            S("Hybrid and remote work",
              P("Office-based employees may work remotely for up to two days per week. Working remotely for more "
                "than two days in a week needs written approval from the department head.", "hr.remote"),
              P("Remote work from outside India is not permitted without prior approval from People Operations "
                "and the Finance team.")),
            S("Attendance recording",
              P("Attendance is recorded in the PeopleHub app. Missed punches must be regularised within five "
                "working days, otherwise the day is treated as unpaid leave.", "hr.attendance")),
        ]),
        Chapter("Leave", [
            S("Annual leave",
              P("Full-time employees accrue 24 working days of paid annual leave per calendar year, credited "
                "monthly.", "hr.annual"),
              P("Up to 10 days of unused annual leave can be carried forward into the next calendar year. Any "
                "balance above 10 days lapses on 31 March.", "hr.carry"),
              P("Leave requests of three days or more should be raised at least two weeks in advance.")),
            S("Sick leave",
              P("Employees receive 12 days of paid sick leave per calendar year. Sick leave does not accrue "
                "beyond the calendar year.", "hr.sick"),
              P("A medical certificate is required for sick leave longer than three consecutive days.",
                "hr.medcert")),
            S("Parental leave",
              P("Birth parents are entitled to 26 weeks of paid parental leave. Other parents, including adoptive "
                "parents, are entitled to 4 weeks of paid parental leave, to be taken within six months of the "
                "birth or adoption.", "hr.parental")),
            S("Public holidays",
              P("Kestrelway Logistics observes 12 paid public holidays per year. The list for each office location "
                "is published on the policy portal every December.", "hr.holidays")),
        ]),
        Chapter("Probation, performance and exit", [
            S("Probation",
              P("New employees serve a probation period of six months. During probation either side may end the "
                "employment with 15 days of notice.", "hr.probation")),
            S("Performance reviews",
              P("Performance reviews take place twice a year, in April and October. Ratings from the April cycle "
                "are used for the annual salary review.", "hr.reviews")),
            S("Notice period",
              P("After confirmation, the notice period is 60 days for individual contributors and 90 days for "
                "people managers.", "hr.notice"),
              P("Unused annual leave may be adjusted against the notice period only with the manager's written "
                "agreement.")),
        ]),
        Chapter("Learning and wellbeing", [
            S("Learning budget",
              P("Each employee has a learning budget of INR 30,000 per financial year for courses, certifications "
                "and books related to their role.", "hr.learning"),
              P("Certification exam fees are reimbursed only after the exam is passed.")),
            S("Employee assistance programme",
              P("The employee assistance programme offers a confidential 24x7 helpline and up to six free "
                "counselling sessions per year for employees and their dependants.", "hr.eap")),
        ]),
    ])


# ------------------------------------------------------------------------------- travel & expense (2 versions)
def travel_policy(v: dict) -> DocSpec:
    return DocSpec(
        file=v["file"], code="KL-FIN-004", title="Travel and Expense Policy", version=v["version"],
        effective=v["effective"], status=v["status"], owner="Finance", supersedes=v["supersedes"],
        superseded_by=v.get("superseded_by", ""),
        chapters=[
            intro("This policy sets out how business travel is approved and booked, and which expenses Kestrelway "
                  "Logistics reimburses.",
                  "It applies to all employees and to contractors whose contract refers to this policy."),
            Chapter("Approval and booking", [
                S("Travel approval",
                  P(f"Every business trip needs approval from the traveller's manager before booking. Trips with an "
                    f"estimated cost above {v['preapproval']} also need approval from the department head.",
                    v["k"] + ".preapproval")),
                S("Air travel",
                  P(f"Flights are booked in economy class. Business class is allowed only for flights with a "
                    f"scheduled duration of more than {v['business_hours']} hours.", v["k"] + ".business"),
                  P(f"Flights must be booked through the travel desk at least {v['advance']} days before departure, "
                    f"unless the trip is urgent and approved as such.", v["k"] + ".advance")),
                S("Rail and road travel",
                  P(f"Use of a personal car for business travel is reimbursed at {v['mileage']} per kilometre. "
                    f"Two-wheelers are reimbursed at {v['mileage2']} per kilometre.", v["k"] + ".mileage"),
                  P("Rail travel is booked in AC 2-tier or the nearest equivalent class.")),
            ]),
            Chapter("Accommodation and meals", [
                S("Hotel and meal limits by city tier",
                  P("Hotel and meal limits depend on the city tier. Tier 1 cities are Mumbai, Delhi NCR, Bengaluru, "
                    "Chennai, Hyderabad, Kolkata and Pune. Tier 2 cities are state capitals not listed in Tier 1. "
                    "All other locations are Tier 3."),
                  T(["City tier", "Hotel cap per night", "Meal allowance per day"],
                    [["Tier 1", v["hotel"][0], v["meal"][0]], ["Tier 2", v["hotel"][1], v["meal"][1]],
                     ["Tier 3", v["hotel"][2], v["meal"][2]]],
                    row_facts={0: v["k"] + ".tier1", 1: v["k"] + ".tier2", 2: v["k"] + ".tier3"}),
                  P("Meal allowances are paid per day of travel and do not require receipts.")),
                S("International travel",
                  P(f"For international trips, a flat per diem of {v['per_diem']} per day covers meals and local "
                    f"transport.", v["k"] + ".perdiem")),
            ]),
            Chapter("Claims and reimbursement", [
                S("Submitting claims",
                  P(f"Expense claims must be submitted in the ExpenseDesk portal within {v['claim_days']} days of "
                    f"the end of the trip. Late claims need approval from the Finance controller.",
                    v["k"] + ".claims"),
                  P(f"An itemised receipt is required for every expense above {v['receipt']}.", v["k"] + ".receipt")),
                S("Payment",
                  P("Approved claims are paid with the next monthly payroll run.", v["k"] + ".payment")),
                S("Non-reimbursable expenses",
                  B(["Alcohol, unless approved in advance for a client event.", "Traffic and parking fines.",
                     "Personal entertainment, including in-room movies.", "Upgrades paid for personal comfort."],
                    v["k"] + ".nonreimb")),
            ]),
            Chapter("Corporate card", [
                S("Card limit and use",
                  P(f"Employees who travel more than four times a year may apply for a corporate card with a monthly "
                    f"limit of {v['card_limit']}.", v["k"] + ".card"),
                  P("The corporate card must not be used for personal spending, even if the amount is repaid.")),
            ]),
        ])


TE_2023 = travel_policy(dict(
    k="te23", file="travel-expense-policy-2023.pdf", version="2023.2", effective="1 July 2023", status="Superseded",
    supersedes="2021.1", superseded_by="2025.1", preapproval="INR 25,000", business_hours="8", advance="10",
    mileage="INR 9", mileage2="INR 4", hotel=["INR 6,000", "INR 4,500", "INR 3,000"],
    meal=["INR 1,500", "INR 1,200", "INR 900"], per_diem="USD 60", claim_days="30", receipt="INR 500",
    card_limit="INR 150,000"))
TE_2025 = travel_policy(dict(
    k="te25", file="travel-expense-policy-2025.pdf", version="2025.1", effective="1 April 2025", status="Current",
    supersedes="2023.2", preapproval="INR 40,000", business_hours="6", advance="14", mileage="INR 12",
    mileage2="INR 5", hotel=["INR 7,500", "INR 5,500", "INR 3,500"], meal=["INR 2,000", "INR 1,600", "INR 1,200"],
    per_diem="USD 75", claim_days="45", receipt="INR 1,000", card_limit="INR 200,000"))


# ------------------------------------------------------------------------------------------- IT security
SEC = DocSpec(
    file="information-security-handbook.pdf", code="KL-IT-002", title="Information Security Handbook",
    version="3.0", effective="15 January 2025", status="Current", owner="IT Security", supersedes="2.4",
    chapters=[
        intro("This handbook describes the security rules every person with a Kestrelway Logistics account must "
              "follow.", "It covers employees, contractors and partner staff who use company accounts or devices."),
        Chapter("Accounts and passwords", [
            S("Password rules",
              P("Passwords must be at least 14 characters long. Passphrases made of several unrelated words are "
                "encouraged.", "sec.pwlen"),
              P("Passwords are not rotated on a schedule. Change your password immediately if you suspect it has "
                "been exposed.", "sec.rotation"),
              P("Use the company-approved password manager to store work credentials.")),
            S("Multi-factor authentication",
              P("Multi-factor authentication is required for email, the VPN and every administrative console.",
                "sec.mfa"),
              P("Authenticator apps or hardware keys are accepted. SMS codes are allowed only as a temporary "
                "fallback for up to seven days.", "sec.sms")),
        ]),
        Chapter("Devices", [
            S("Laptop security",
              P("All company laptops use full-disk encryption, which is enabled before the device is issued.",
                "sec.encrypt"),
              P("Screens lock automatically after 5 minutes of inactivity.", "sec.lock")),
            S("Removable media",
              P("USB storage devices are blocked by default. Exceptions are granted for a business need through an "
                "IT service desk ticket and expire after 30 days.", "sec.usb")),
            S("Patching",
              P("Operating system and browser updates must be installed within 14 days of release. Critical "
                "security patches must be installed within 72 hours.", "sec.patch")),
        ]),
        Chapter("Incidents", [
            S("Reporting a security incident",
              P("Report any suspected security incident to the Security Desk within 1 hour of noticing it, through "
                "the #sec-incidents channel or by email to security@kestrelway.example.", "sec.incident"),
              P("Do not try to investigate an incident yourself and do not delete suspicious emails or files.")),
            S("Phishing",
              P("Use the Report Phishing button in the mail client to report suspicious emails. The Security Desk "
                "reviews each report and blocks the sender if needed.", "sec.phish")),
            S("Lost or stolen devices",
              P("A lost or stolen laptop or phone must be reported within 1 hour so that it can be wiped remotely.",
                "sec.lost")),
        ]),
        Chapter("Data classification", [
            S("Classification levels",
              P("Every document and dataset is labelled with one of four classification levels."),
              T(["Level", "Examples", "Handling rule"],
                [["Public", "Press releases, published tariffs", "No restrictions"],
                 ["Internal", "Policies, org charts", "Share only with staff and partners"],
                 ["Confidential", "Customer contracts, pricing", "Need-to-know access, encrypted in transit"],
                 ["Restricted", "Payroll, identity documents", "Owner-approved access, encrypted at rest"]],
                row_facts={3: "sec.restricted", 2: "sec.confidential"})),
            S("Remote access",
              P("Internal systems are reachable only through the company VPN. Public Wi-Fi may be used only with the "
                "VPN connected.", "sec.vpn")),
        ]),
    ])


# --------------------------------------------------------------------------------------- product manual
MANUAL = DocSpec(
    file="routebox-r2-user-manual.pdf", code="KL-PRD-010", title="RouteBox R2 Telematics Unit User Manual",
    version="1.4", effective="10 February 2025", status="Current", owner="Product Engineering", supersedes="1.3",
    chapters=[
        intro("This manual explains how to install, operate and maintain the RouteBox R2 telematics unit fitted to "
              "Kestrelway Logistics vehicles.",
              "It is written for fleet technicians and depot supervisors. Read the safety notes before installing "
              "the unit."),
        Chapter("Specifications", [
            S("Technical data",
              T(["Parameter", "Value"],
                [["Operating voltage", "9 to 36 V DC"], ["Backup battery", "1,800 mAh, up to 8 hours of tracking"],
                 ["Ingress protection", "IP67"], ["Operating temperature", "-20 to +70 degrees Celsius"],
                 ["Positioning", "GPS, GLONASS and Galileo, 2.5 m accuracy"],
                 ["Cellular", "4G LTE Cat-1 with 2G fallback"], ["Dimensions", "98 x 64 x 28 mm"],
                 ["Weight", "165 g"]],
                row_facts={0: "rb.voltage", 1: "rb.battery", 2: "rb.ip", 3: "rb.temp", 4: "rb.gnss",
                           5: "rb.cell", 7: "rb.weight"})),
            S("Reporting interval",
              P("By default the unit reports its position every 30 seconds while the ignition is on. The interval "
                "can be configured between 10 seconds and 10 minutes from the fleet console.", "rb.interval")),
        ]),
        Chapter("Installation", [
            S("Mounting",
              P("Mount the unit with the label facing up and a clear view of the sky, for example under the "
                "dashboard top cover. Keep it at least 20 cm away from radio transmitters.", "rb.mount")),
            S("Wiring",
              B(["Connect the red wire to permanent 12 or 24 V supply through a 2 A inline fuse.",
                 "Connect the black wire to vehicle ground.",
                 "Connect the yellow wire to the ignition (ACC) line so the unit can detect trips.",
                 "Leave the white wire unconnected unless a door sensor is fitted."], "rb.wiring")),
        ]),
        Chapter("Status lights", [
            S("LED codes",
              T(["LED", "Meaning"],
                [["Green, steady", "Position fix and network connection are OK"],
                 ["Green, blinking", "Searching for satellites"],
                 ["Amber, blinking", "No cellular network; data is buffered"],
                 ["Red, blinking", "Backup battery low"],
                 ["Red, steady", "Hardware fault; contact support"]],
                row_facts={1: "rb.led_green_blink", 2: "rb.led_amber", 3: "rb.led_red_blink", 4: "rb.led_red"})),
        ]),
        Chapter("Firmware and maintenance", [
            S("Firmware updates",
              P("Firmware updates are installed over the air between 01:00 and 04:00 local time while the ignition "
                "is off.", "rb.ota"),
              P("A manual update over USB-C with the RouteBox Updater takes about 6 minutes. Do not disconnect power "
                "during an update.", "rb.manual_update")),
            S("Factory reset",
              P("To restore factory settings, hold the service button for 15 seconds until all lights flash.",
                "rb.reset")),
            S("Warranty",
              P("The RouteBox R2 has a warranty of 24 months from the installation date. Damage from incorrect "
                "wiring is not covered.", "rb.warranty")),
        ]),
    ])


# ----------------------------------------------------------------------------------------- SLA (2 versions)
def sla(v: dict) -> DocSpec:
    return DocSpec(
        file=v["file"], code="KL-LEG-007", title="Kestrelway Track Service Level Agreement", version=v["version"],
        effective=v["effective"], status=v["status"], owner="Legal and Customer Success", supersedes=v["supersedes"],
        superseded_by=v.get("superseded_by", ""),
        chapters=[
            intro("This agreement defines the service levels for Kestrelway Track, the shipment tracking platform "
                  "offered to business customers.",
                  "It applies to all customers on a paid Kestrelway Track subscription. Trial accounts are excluded."),
            Chapter("Availability", [
                S("Uptime commitment",
                  P(f"Kestrelway Track will be available at least {v['uptime']} of the time in each calendar month, "
                    f"excluding scheduled maintenance.", v["k"] + ".uptime"),
                  P(f"Availability is measured by external probes that check the platform {v['probe']} from three "
                    f"regions.", v["k"] + ".probe")),
                S("Definitions",
                  P("Downtime means a period in which the tracking API or web console returns errors for more than "
                    "half of the probe checks. Scheduled maintenance is not downtime.")),
            ]),
            Chapter("Service credits", [
                S("Credit levels",
                  P("If monthly availability falls below the commitment, the customer receives a service credit on "
                    "the next invoice."),
                  T(["Monthly availability", "Service credit"], v["credits"], row_facts=v["credit_facts"]),
                  P("Service credits are capped at 50% of the monthly fee and are the customer's only remedy for "
                    "missed availability.", v["k"] + ".cap")),
                S("Claiming credits",
                  P(f"Customers must request service credits within {v['claim_days']} days after the end of the "
                    f"affected month.", v["k"] + ".claim")),
            ]),
            Chapter("Support", [
                S("Response times",
                  T(["Priority", "Example", "First response", "Target resolution"], v["support"],
                    row_facts={0: v["k"] + ".p1", 1: v["k"] + ".p2", 2: v["k"] + ".p3"}),
                  P(f"P1 incidents are handled 24x7. Other priorities are handled during support hours, "
                    f"{v['hours']}.", v["k"] + ".hours")),
            ]),
            Chapter("Maintenance and data protection", [
                S("Scheduled maintenance",
                  P(f"Scheduled maintenance takes place on {v['window']}. Customers are notified at least "
                    f"{v['notice']} hours in advance.", v["k"] + ".maint")),
                S("Backups and recovery",
                  P(f"Customer data is backed up daily and backups are kept for {v['backup_days']} days. The recovery "
                    f"point objective is {v['rpo']} and the recovery time objective is {v['rto']}.",
                    v["k"] + ".backup")),
            ]),
        ])


SLA_2024 = sla(dict(
    k="sla24", file="track-sla-2024.pdf", version="1.0", effective="1 January 2024", status="Superseded",
    supersedes="None", superseded_by="2.0", uptime="99.5%", probe="every 5 minutes",
    credits=[["99.0% to below 99.5%", "5% of the monthly fee"], ["95.0% to below 99.0%", "15% of the monthly fee"],
             ["Below 95.0%", "30% of the monthly fee"]],
    credit_facts={0: "sla24.credit1", 2: "sla24.credit3"}, claim_days="15",
    support=[["P1", "Platform down for all users", "1 hour", "8 hours"],
             ["P2", "Major feature unavailable", "4 hours", "2 business days"],
             ["P3", "Minor issue or question", "1 business day", "10 business days"]],
    hours="09:00 to 18:00 IST, Monday to Friday", window="Saturdays from 22:00 to 02:00 IST", notice="48",
    backup_days="14", rpo="4 hours", rto="12 hours"))
SLA_2025 = sla(dict(
    k="sla25", file="track-sla-2025.pdf", version="2.0", effective="1 July 2025", status="Current",
    supersedes="1.0", uptime="99.9%", probe="every minute",
    credits=[["99.0% to below 99.9%", "10% of the monthly fee"], ["95.0% to below 99.0%", "25% of the monthly fee"],
             ["Below 95.0%", "50% of the monthly fee"]],
    credit_facts={0: "sla25.credit1", 2: "sla25.credit3"}, claim_days="30",
    support=[["P1", "Platform down for all users", "30 minutes", "4 hours"],
             ["P2", "Major feature unavailable", "2 hours", "1 business day"],
             ["P3", "Minor issue or question", "8 business hours", "5 business days"]],
    hours="08:00 to 20:00 IST, Monday to Saturday", window="Sundays from 02:00 to 05:00 IST", notice="72",
    backup_days="35", rpo="1 hour", rto="4 hours"))


# ------------------------------------------------------------------------------------------ warehouse safety
SAFETY = DocSpec(
    file="warehouse-safety-procedures.pdf", code="KL-OPS-003", title="Warehouse Safety Procedures", version="2.1",
    effective="1 March 2025", status="Current", owner="Operations Safety", supersedes="2.0",
    chapters=[
        intro("These procedures set the minimum safety rules for all Kestrelway Logistics warehouses and "
              "cross-dock sites.",
              "They apply to employees, agency workers, drivers and visitors while they are on a warehouse floor "
              "or yard."),
        Chapter("Personal protective equipment", [
            S("Required equipment",
              P("Safety shoes and high-visibility vests are mandatory for everyone on the warehouse floor.",
                "ops.ppe"),
              P("Hard hats are required in high-bay racking zones. Cut-resistant gloves must be worn when opening "
                "cartons with a blade.", "ops.hardhat")),
        ]),
        Chapter("Material handling equipment", [
            S("Forklifts",
              P("The forklift speed limit is 8 km/h inside buildings and 15 km/h in the yard.", "ops.speed"),
              P("Forklift operators need a valid licence and an internal certification that is renewed every 12 "
                "months.", "ops.cert"),
              P("Operators complete the pre-shift checklist in the SafeLog app before the first lift of each "
                "shift.", "ops.checklist")),
            S("Stacking",
              P("Pallets stored on the floor may be stacked at most three high, and never in front of fire exits "
                "or extinguishers.", "ops.stack")),
        ]),
        Chapter("Manual handling", [
            S("Lifting limits",
              P("The individual lifting limit is 25 kg. Heavier items must be moved as a team lift or with "
                "equipment.", "ops.lift"),
              P("Use the lifting aids provided at each packing station and rotate tasks every two hours.",
                "ops.rotate")),
        ]),
        Chapter("Emergencies and reporting", [
            S("Fire safety",
              P("Fire drills are held once every quarter at each site. The assembly point is the Gate 2 car park.",
                "ops.drill")),
            S("First aid",
              P("Each shift must have at least one trained first aider for every 50 people on site.", "ops.firstaid")),
            S("Incident and near-miss reporting",
              P("All injuries, incidents and near misses must be recorded in SafeLog within 24 hours.",
                "ops.incident")),
            S("Lithium batteries",
              P("Lithium batteries and battery-powered goods are stored in Zone H, which has a separate spill kit "
                "and a fire-resistant cabinet.", "ops.lithium")),
        ]),
    ])


# --------------------------------------------------------------------------------------- claims & returns
CLAIMS = DocSpec(
    file="freight-claims-returns-policy.pdf", code="KL-CS-005", title="Freight Claims and Returns Policy",
    version="1.2", effective="1 May 2025", status="Current", owner="Customer Service", supersedes="1.1",
    chapters=[
        intro("This policy explains how customers report damaged or lost freight and how return shipments are "
              "handled.", "It applies to domestic road freight and parcel shipments booked with Kestrelway "
              "Logistics."),
        Chapter("Damage and loss", [
            S("Reporting deadlines",
              P("Visible damage must be noted on the delivery note at the time of delivery.", "cs.visible"),
              P("Concealed damage must be reported within 7 days of delivery, with photos of the packaging.",
                "cs.concealed"),
              P("Claims for lost shipments must be filed within 30 days of the expected delivery date.", "cs.lost")),
        ]),
        Chapter("Liability", [
            S("Limits of liability",
              P("Unless the shipment is insured, liability is limited to INR 500 per kilogram of gross weight.",
                "cs.liability"),
              P("Optional cargo insurance costs 0.35% of the declared value and covers the full declared value.",
                "cs.insurance")),
            S("Exclusions",
              B(["Perishable goods shipped without a temperature-controlled booking.",
                 "Damage caused by inadequate packaging by the shipper.",
                 "Delays caused by customs inspection or road closures."], "cs.exclusions")),
        ]),
        Chapter("Claim handling", [
            S("Decision and payment",
              P("A claim decision is made within 21 days after all documents are received.", "cs.decision"),
              P("Approved claims are paid within 10 business days of the decision.", "cs.payment")),
            S("Documents required",
              B(["Copy of the invoice for the goods.", "Delivery note with the damage remark, if any.",
                 "Photos of the goods and packaging.", "Claim form signed by the consignee."], "cs.documents")),
        ]),
        Chapter("Returns", [
            S("Return pickups",
              P("Return pickups booked before 15:00 are collected on the next business day.", "cs.pickup"),
              P("A return label stays valid for 14 days after it is issued.", "cs.label")),
        ]),
    ])


DOCS: list[DocSpec] = [HR, TE_2023, TE_2025, SEC, MANUAL, SLA_2024, SLA_2025, SAFETY, CLAIMS]


# -------------------------------------------------------------------------------------------- gold QA set
QAS: list[QA] = [
    # HR
    QA("hr01", "How many hours make up the standard working week?", "hr.week", "40 hours"),
    QA("hr02", "What are the core hours employees must be available?", "hr.core", "10:00 to 16:00 IST"),
    QA("hr03", "How many days a week can I work from home?", "hr.remote", "two days per week"),
    QA("hr04", "How long do I have to fix a missed attendance punch?", "hr.attendance", "five working days"),
    QA("hr05", "How much paid annual leave do full-time staff get?", "hr.annual", "24 working days"),
    QA("hr06", "How many unused vacation days can roll over to next year?", "hr.carry", "10 days"),
    QA("hr07", "How many paid sick days per year?", "hr.sick", "12 days"),
    QA("hr08", "When do I need a doctor's note for being ill?", "hr.medcert", "longer than three consecutive days"),
    QA("hr09", "How long is maternity leave for the birth parent?", "hr.parental", "26 weeks"),
    QA("hr10", "How many public holidays does the company give?", "hr.holidays", "12 paid public holidays"),
    QA("hr11", "How long is the probation period for new joiners?", "hr.probation", "six months"),
    QA("hr12", "In which months are performance reviews held?", "hr.reviews", "April and October"),
    QA("hr13", "What is the notice period for a people manager?", "hr.notice", "90 days"),
    QA("hr14", "What is the yearly training budget per employee?", "hr.learning", "INR 30,000"),
    QA("hr15", "How many free counselling sessions does the EAP include?", "hr.eap", "six free counselling sessions"),
    # Travel & expense, current version (no year in the question -> the current policy)
    QA("te01", "Above what trip cost is department head approval needed?", "te25.preapproval", "INR 40,000"),
    QA("te02", "When am I allowed to fly business class?", "te25.business", "more than 6 hours"),
    QA("te03", "How far in advance must flights be booked through the travel desk?", "te25.advance", "14 days"),
    QA("te04", "What is the mileage rate for using my own car?", "te25.mileage", "INR 12"),
    QA("te05", "What is the hotel limit per night in a Tier 1 city?", "te25.tier1", "INR 7,500"),
    QA("te06", "What daily meal allowance applies in Tier 2 cities?", "te25.tier2", "INR 1,600"),
    QA("te07", "What is the per diem for international trips?", "te25.perdiem", "USD 75"),
    QA("te08", "What is the deadline for submitting an expense claim?", "te25.claims", "45 days"),
    QA("te09", "Above what amount do I need an itemised receipt?", "te25.receipt", "INR 1,000"),
    QA("te10", "What is the monthly limit on the corporate card?", "te25.card", "INR 200,000"),
    QA("te11", "When are approved expense claims paid out?", "te25.payment", "next monthly payroll run"),
    QA("te12", "Are parking fines reimbursed?", "te25.nonreimb", "Traffic and parking fines"),
    # Travel & expense, explicitly the 2023 version
    QA("te13", "Under the 2023 travel policy, what was the mileage rate for a personal car?", "te23.mileage", "INR 9"),
    QA("te14", "What was the Tier 1 hotel cap per night in the 2023 travel and expense policy?", "te23.tier1",
       "INR 6,000"),
    QA("te15", "How many days did the 2023 expense policy allow for submitting claims?", "te23.claims", "30 days"),
    QA("te16", "What was the international per diem in the 2023 version of the travel policy?", "te23.perdiem",
       "USD 60"),
    # IT security
    QA("se01", "What is the minimum password length?", "sec.pwlen", "14 characters"),
    QA("se02", "Do I have to change my password every 90 days?", "sec.rotation", "not rotated on a schedule"),
    QA("se03", "Which systems require MFA?", "sec.mfa", "email, the VPN and every administrative console"),
    QA("se04", "Can I use text message codes for two-factor login?", "sec.sms", "temporary fallback"),
    QA("se05", "After how long does the laptop screen lock itself?", "sec.lock", "5 minutes"),
    QA("se06", "Can I use a USB stick on my work laptop?", "sec.usb", "blocked by default"),
    QA("se07", "How quickly must critical security patches be installed?", "sec.patch", "72 hours"),
    QA("se08", "How soon must I report a suspected security incident?", "sec.incident", "within 1 hour"),
    QA("se09", "How do I report a phishing email?", "sec.phish", "Report Phishing button"),
    QA("se10", "How should payroll data be handled?", "sec.restricted", "Owner-approved access, encrypted at rest"),
    QA("se11", "Can I use public Wi-Fi for work?", "sec.vpn", "only with the VPN connected"),
    QA("se12", "Are company laptops encrypted?", "sec.encrypt", "full-disk encryption"),
    # RouteBox R2 manual
    QA("rb01", "What supply voltage range does the RouteBox R2 accept?", "rb.voltage", "9 to 36 V DC"),
    QA("rb02", "How long does the RouteBox backup battery last?", "rb.battery", "up to 8 hours"),
    QA("rb03", "What is the ingress protection rating of the R2 unit?", "rb.ip", "IP67"),
    QA("rb04", "What temperature range can the telematics unit operate in?", "rb.temp",
       "-20 to +70 degrees Celsius"),
    QA("rb05", "How often does the device send its location by default?", "rb.interval", "every 30 seconds"),
    QA("rb06", "Which wire goes to the ignition line?", "rb.wiring", "yellow wire"),
    QA("rb07", "What does a blinking amber LED mean on the RouteBox?", "rb.led_amber", "No cellular network"),
    QA("rb08", "What does a steady red light indicate?", "rb.led_red", "Hardware fault"),
    QA("rb09", "When are over-the-air firmware updates installed?", "rb.ota", "between 01:00 and 04:00"),
    QA("rb10", "How do I factory reset the RouteBox R2?", "rb.reset", "15 seconds"),
    QA("rb11", "How long is the RouteBox warranty?", "rb.warranty", "24 months"),
    QA("rb12", "How much does the RouteBox R2 weigh?", "rb.weight", "165 g"),
    # SLA, current version
    QA("sl01", "What uptime does the Kestrelway Track SLA guarantee?", "sla25.uptime", "99.9%"),
    QA("sl02", "What is the first response time for a P1 incident?", "sla25.p1", "30 minutes"),
    QA("sl03", "What is the target resolution time for P2 tickets?", "sla25.p2", "1 business day"),
    QA("sl04", "What service credit applies if availability drops below 95%?", "sla25.credit3",
       "50% of the monthly fee"),
    QA("sl05", "Within how many days must a customer claim service credits?", "sla25.claim", "30 days"),
    QA("sl06", "When is the scheduled maintenance window for Kestrelway Track?", "sla25.maint",
       "Sundays from 02:00 to 05:00 IST"),
    QA("sl07", "How long are customer data backups retained?", "sla25.backup", "35 days"),
    QA("sl08", "What are the support hours for non-P1 tickets?", "sla25.hours",
       "08:00 to 20:00 IST, Monday to Saturday"),
    # SLA, explicitly version 1.0 / 2024
    QA("sl09", "What uptime did the 2024 SLA (version 1.0) commit to?", "sla24.uptime", "99.5%"),
    QA("sl10", "In the 2024 SLA, what was the P1 first response time?", "sla24.p1", "1 hour"),
    # Warehouse safety
    QA("op01", "What is the forklift speed limit indoors?", "ops.speed", "8 km/h"),
    QA("op02", "How often must forklift certification be renewed?", "ops.cert", "every 12 months"),
    QA("op03", "What is the maximum weight one person may lift?", "ops.lift", "25 kg"),
    QA("op04", "How high can pallets be stacked on the floor?", "ops.stack", "three high"),
    QA("op05", "How often are fire drills held?", "ops.drill", "once every quarter"),
    QA("op06", "Where is the fire assembly point?", "ops.drill", "Gate 2 car park"),
    QA("op07", "How many first aiders are needed per shift?", "ops.firstaid", "every 50 people"),
    QA("op08", "Within what time must a near miss be logged?", "ops.incident", "24 hours"),
    QA("op09", "Where are lithium batteries stored?", "ops.lithium", "Zone H"),
    QA("op10", "What PPE is mandatory on the warehouse floor?", "ops.ppe",
       "Safety shoes and high-visibility vests"),
    # Claims and returns
    QA("cs01", "How long does a customer have to report concealed damage?", "cs.concealed", "7 days"),
    QA("cs02", "What is the deadline to claim for a lost shipment?", "cs.lost", "30 days"),
    QA("cs03", "What is the liability limit per kilogram for uninsured freight?", "cs.liability", "INR 500"),
    QA("cs04", "How much does optional cargo insurance cost?", "cs.insurance", "0.35% of the declared value"),
    QA("cs05", "How quickly is a freight claim decided?", "cs.decision", "21 days"),
    QA("cs06", "How long is a return label valid?", "cs.label", "14 days"),
    QA("cs07", "What is the cut-off time for next-day return pickup?", "cs.pickup", "before 15:00"),
    # Unanswerable: the corpus does not contain the answer (several are close to real topics on purpose)
    QA("na01", "What is the dental insurance coverage limit for employees?", None),
    QA("na02", "How many days of paid leave do contractors in Germany receive?", None),
    QA("na03", "What is the price of a Kestrelway Track enterprise licence?", None),
    QA("na04", "Who is the CEO of Kestrelway Logistics?", None),
    QA("na05", "What is the reimbursement rate for charging an electric vehicle?", None),
    QA("na06", "How long are warehouse CCTV recordings kept?", None),
    QA("na07", "What bonus percentage do warehouse supervisors receive?", None),
    QA("na08", "Which cloud provider hosts Kestrelway Track?", None),
    QA("na09", "What is the maximum parcel weight for same-day delivery?", None),
    QA("na10", "What is the overtime pay rate for forklift operators?", None),
    QA("na11", "Are pets allowed in the office?", None),
    QA("na12", "Does the RouteBox R2 support satellite messaging when there is no cellular coverage?", None),
    QA("na13", "What is the dress code for client meetings?", None),
    QA("na14", "How many electric trucks are in the Kestrelway fleet?", None),
]
