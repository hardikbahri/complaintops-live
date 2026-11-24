"""Turn a CFPB complaint's product + issue into a ComplaintOps class label.

Same mapping as the training data (complaintops/data.py), so live accuracy is measured
against the label the model was trained to predict. Labels are then merged to the 24-class
taxonomy, exactly as the served model does.
"""

PRODUCT_MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit card or prepaid card": "Credit card",
    "Prepaid card": "Credit card",
    "Bank account or service": "Checking or savings account",
    "Payday loan, title loan, or personal loan": "Payday/personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday/personal loan",
    "Payday loan": "Payday/personal loan",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
}

ISSUE_MAP = {
    "Problem with a company's investigation into an existing problem":
        "Problem with a credit reporting company's investigation into an existing problem",
    "Incorrect information on credit report": "Incorrect information on your report",
    "Dealing with my lender or servicer": "Dealing with your lender or servicer",
    "Cont'd attempts collect debt not owed": "Attempts to collect debt not owed",
}

_P = "Credit reporting | "
MERGED_LABEL = _P + "Dispute, accuracy or use of report"
MERGE_MAP = {
    _P + "Incorrect information on your report": MERGED_LABEL,
    _P + "Problem with a credit reporting company's investigation into an existing problem": MERGED_LABEL,
    _P + "Improper use of your report": MERGED_LABEL,
}


def merge_label(label):
    return MERGE_MAP.get(label, label)


def merged_classes(classes):
    out = []
    for c in classes:
        m = merge_label(c)
        if m not in out:
            out.append(m)
    return out


def true_label(product, issue):
    product = PRODUCT_MAP.get(product or "", product or "")
    issue = ISSUE_MAP.get(issue or "", issue or "")
    return merge_label(f"{product} | {issue}")
