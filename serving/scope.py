"""Real-estate scoping (B-15): the curated home-relevant subset of the 67 MIT
Indoor classes. A prediction in this subset is surfaced as a room tag; anything
else is routed to the review queue as out-of-scope.

Single source of truth: the API, the frontend's correction dropdown (via
GET /classes), the scoped retrain (M3) and the field evaluation all read this,
so the definition of "in scope" cannot drift between them.

Keys are raw MIT class dirs; values are display labels for the UI.
"""

HOME_CLASS_LABELS = {
    "artstudio":     "Art studio",
    "bar":           "Bar / lounge",
    "bathroom":      "Bathroom",
    "bedroom":       "Bedroom",
    "children_room": "Children's room",
    "closet":        "Closet",
    "corridor":      "Hallway",
    "dining_room":   "Dining room",
    "gameroom":      "Game room",
    "garage":        "Garage",
    "greenhouse":    "Greenhouse / sunroom",
    "gym":           "Home gym",
    "kitchen":       "Kitchen",
    "laundromat":    "Laundry room",
    "library":       "Home library",
    "livingroom":    "Living room",
    "lobby":         "Lobby / entryway",
    "nursery":       "Nursery",
    "office":        "Home office",
    "pantry":        "Pantry",
    "poolinside":    "Indoor pool",
    "stairscase":    "Staircase",
    "studiomusic":   "Studio",
    "winecellar":    "Wine cellar",
}

# Class name of the catch-all used by the scoped (24 + other) model.
OTHER_CLASS = "other"
