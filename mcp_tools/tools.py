import uuid


def create_support_ticket(issue_description, ticket_store):

    ticket_id = str(uuid.uuid4())[:8]

    ticket_store[ticket_id] = {
        "issue": issue_description,
        "status": "Open"
    }

    return {
        "ticket_id": ticket_id,
        "status": "Open",
        "message": "Support ticket created successfully."
    }


def check_ticket_status(ticket_id, ticket_store):

    if ticket_id in ticket_store:
        return ticket_store[ticket_id]

    return {
        "error": "Ticket not found"
    }
