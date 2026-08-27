#!/usr/bin/env python3
"""
Generate synthetic San Francisco real estate listings for development and testing.
Produces two CSV files in data/:
  - listings.csv        (2000 listings)
  - listing_photos.csv  (5-10 photos per listing, one marked main_photo=True)

Photo column contains filename references (e.g. SF00001_01.jpg).
Actual image files are not generated; swap in real images from your MLS feed.
"""

import csv
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Geography & neighborhood definitions
# ---------------------------------------------------------------------------

NEIGHBORHOODS = [
    {"name": "Mission District",   "price_mult": 0.92, "w": 85,
     "types": {"Condo": 0.40, "Single Family": 0.22, "Townhouse": 0.18, "Multi Family": 0.20},
     "streets": ["Valencia St", "Mission St", "Guerrero St", "Dolores St", "24th St", "20th St", "Army St", "Cesar Chavez St"],
     "character": "vibrant, diverse neighborhood with taquerias, bars, and murals",
     "elem": ["Cesar Chavez Elementary", "Fairmount Elementary", "Daniel Webster Elementary"],
     "middle": ["James Lick Middle School", "Everett Middle School"],
     "high": ["Mission High School", "John O'Connell High School"]},

    {"name": "Noe Valley",         "price_mult": 1.28, "w": 75,
     "types": {"Condo": 0.30, "Single Family": 0.42, "Townhouse": 0.20, "Multi Family": 0.08},
     "streets": ["24th St", "Sanchez St", "Church St", "Vicksburg St", "Elizabeth St", "Noe St", "Clipper St"],
     "character": "sunny, family-friendly village feel with boutiques and cafés",
     "elem": ["Alvarado Elementary", "Douglass Elementary", "James Lick Elementary"],
     "middle": ["James Lick Middle School", "Aptos Middle School"],
     "high": ["Abraham Lincoln High School", "Wallenberg Traditional High School"]},

    {"name": "Pacific Heights",    "price_mult": 1.75, "w": 60,
     "types": {"Condo": 0.45, "Single Family": 0.35, "Townhouse": 0.15, "Multi Family": 0.05},
     "streets": ["Broadway", "Vallejo St", "Jackson St", "Washington St", "Divisadero St", "Fillmore St", "Lyon St"],
     "character": "prestigious neighborhood with grand Victorians and sweeping bay views",
     "elem": ["McKinley Elementary", "Sherman Elementary", "Argonne Elementary"],
     "middle": ["Presidio Middle School", "Marina Middle School"],
     "high": ["Galileo High School", "George Washington High School"]},

    {"name": "SOMA",               "price_mult": 0.88, "w": 90,
     "types": {"Condo": 0.72, "Single Family": 0.03, "Townhouse": 0.15, "Multi Family": 0.10},
     "streets": ["Folsom St", "Howard St", "Brannan St", "King St", "2nd St", "3rd St", "4th St", "Townsend St"],
     "character": "urban tech hub with sleek high-rises and proximity to the waterfront",
     "elem": ["Bessie Carmichael Elementary", "Mira Loma Elementary"],
     "middle": ["Francisco Middle School", "Hoover Middle School"],
     "high": ["Galileo High School", "City Arts & Tech High School"]},

    {"name": "Castro",             "price_mult": 1.15, "w": 65,
     "types": {"Condo": 0.50, "Single Family": 0.28, "Townhouse": 0.14, "Multi Family": 0.08},
     "streets": ["Castro St", "18th St", "Market St", "Noe St", "Sanchez St", "Liberty St", "States St"],
     "character": "iconic, walkable neighborhood with strong community character",
     "elem": ["Harvey Milk Civil Rights Academy", "Sanchez Elementary"],
     "middle": ["James Lick Middle School", "Everett Middle School"],
     "high": ["Mission High School", "Abraham Lincoln High School"]},

    {"name": "Haight-Ashbury",     "price_mult": 1.05, "w": 55,
     "types": {"Condo": 0.38, "Single Family": 0.35, "Townhouse": 0.12, "Multi Family": 0.15},
     "streets": ["Haight St", "Ashbury St", "Clayton St", "Cole St", "Masonic Ave", "Page St", "Waller St"],
     "character": "eclectic Victorian neighborhood steps from Golden Gate Park",
     "elem": ["Grattan Elementary", "Clarendon Elementary", "Harvey Milk Civil Rights Academy"],
     "middle": ["Presidio Middle School", "Aptos Middle School"],
     "high": ["Abraham Lincoln High School", "Galileo High School"]},

    {"name": "Inner Richmond",     "price_mult": 1.10, "w": 65,
     "types": {"Condo": 0.42, "Single Family": 0.33, "Townhouse": 0.15, "Multi Family": 0.10},
     "streets": ["Clement St", "Geary Blvd", "California St", "6th Ave", "8th Ave", "10th Ave", "Arguello Blvd"],
     "character": "diverse, quiet neighborhood with authentic dining and proximity to the park",
     "elem": ["Lafayette Elementary", "Argonne Elementary", "Lone Mountain Elementary"],
     "middle": ["Presidio Middle School", "Roosevelt Middle School"],
     "high": ["Galileo High School", "George Washington High School"]},

    {"name": "Outer Richmond",     "price_mult": 0.98, "w": 60,
     "types": {"Condo": 0.35, "Single Family": 0.40, "Townhouse": 0.15, "Multi Family": 0.10},
     "streets": ["Balboa St", "Cabrillo St", "Fulton St", "Point Lobos Ave", "38th Ave", "42nd Ave", "48th Ave"],
     "character": "quiet, foggy residential neighborhood near Ocean Beach",
     "elem": ["Francis Scott Key Elementary", "Sutro Elementary", "Dianne Feinstein Elementary"],
     "middle": ["Roosevelt Middle School", "Hoover Middle School"],
     "high": ["George Washington High School", "Abraham Lincoln High School"]},

    {"name": "Inner Sunset",       "price_mult": 1.08, "w": 65,
     "types": {"Condo": 0.40, "Single Family": 0.38, "Townhouse": 0.14, "Multi Family": 0.08},
     "streets": ["Irving St", "Judah St", "9th Ave", "7th Ave", "Lincoln Way", "Kirkham St", "Lawton St"],
     "character": "laid-back neighborhood with great restaurants and quick park access",
     "elem": ["Inner Sunset Elementary", "Grattan Elementary", "Clarendon Elementary"],
     "middle": ["Aptos Middle School", "Presidio Middle School"],
     "high": ["Abraham Lincoln High School", "Galileo High School"]},

    {"name": "Outer Sunset",       "price_mult": 0.93, "w": 60,
     "types": {"Condo": 0.30, "Single Family": 0.48, "Townhouse": 0.12, "Multi Family": 0.10},
     "streets": ["Noriega St", "Taraval St", "Judah St", "32nd Ave", "39th Ave", "45th Ave", "Sunset Blvd"],
     "character": "relaxed, breezy neighborhood beloved for its ocean proximity and surf culture",
     "elem": ["Lawton Elementary", "Sunset Elementary", "McCoppin Elementary"],
     "middle": ["Aptos Middle School", "Hoover Middle School"],
     "high": ["Abraham Lincoln High School", "George Washington High School"]},

    {"name": "Marina",             "price_mult": 1.35, "w": 65,
     "types": {"Condo": 0.55, "Single Family": 0.20, "Townhouse": 0.18, "Multi Family": 0.07},
     "streets": ["Chestnut St", "Union St", "Marina Blvd", "Lombard St", "Divisadero St", "Cervantes Blvd", "Alhambra St"],
     "character": "lively, upscale neighborhood with waterfront access and boutique shopping",
     "elem": ["Moscone Elementary", "Sherman Elementary", "Winfield Scott Elementary"],
     "middle": ["Marina Middle School", "Presidio Middle School"],
     "high": ["Galileo High School", "George Washington High School"]},

    {"name": "North Beach",        "price_mult": 1.22, "w": 50,
     "types": {"Condo": 0.62, "Single Family": 0.15, "Townhouse": 0.15, "Multi Family": 0.08},
     "streets": ["Columbus Ave", "Grant Ave", "Vallejo St", "Green St", "Filbert St", "Union St", "Kearny St"],
     "character": "historic Italian neighborhood with celebrated cafés and steps to Washington Square Park",
     "elem": ["Jean Parker Elementary", "Garfield Elementary"],
     "middle": ["Francisco Middle School", "Marina Middle School"],
     "high": ["Galileo High School", "City Arts & Tech High School"]},

    {"name": "Bernal Heights",     "price_mult": 1.05, "w": 65,
     "types": {"Condo": 0.32, "Single Family": 0.42, "Townhouse": 0.16, "Multi Family": 0.10},
     "streets": ["Cortland Ave", "Mission St", "Andover St", "Folsom St", "Precita Ave", "Bernal Heights Blvd", "Elsie St"],
     "character": "tight-knit hilltop community with stunning panoramic views",
     "elem": ["Fairmount Elementary", "Cesar Chavez Elementary", "Esther Clark Elementary"],
     "middle": ["James Lick Middle School", "Everett Middle School"],
     "high": ["John O'Connell High School", "Mission High School"]},

    {"name": "Potrero Hill",       "price_mult": 1.12, "w": 55,
     "types": {"Condo": 0.42, "Single Family": 0.32, "Townhouse": 0.18, "Multi Family": 0.08},
     "streets": ["18th St", "20th St", "Connecticut St", "Texas St", "Wisconsin St", "Arkansas St", "De Haro St"],
     "character": "sunny micro-climate hill with artsy studios and city skyline views",
     "elem": ["Daniel Webster Elementary", "Bryant Elementary"],
     "middle": ["Hoover Middle School", "James Lick Middle School"],
     "high": ["John O'Connell High School", "Mission High School"]},

    {"name": "Hayes Valley",       "price_mult": 1.18, "w": 70,
     "types": {"Condo": 0.58, "Single Family": 0.18, "Townhouse": 0.16, "Multi Family": 0.08},
     "streets": ["Hayes St", "Octavia Blvd", "Laguna St", "Gough St", "Fell St", "Oak St", "Grove St"],
     "character": "chic, walkable neighborhood with designer boutiques and top-rated restaurants",
     "elem": ["Buena Vista Elementary", "John Muir Elementary"],
     "middle": ["Everett Middle School", "James Lick Middle School"],
     "high": ["Mission High School", "Galileo High School"]},

    {"name": "Dogpatch",           "price_mult": 1.05, "w": 50,
     "types": {"Condo": 0.55, "Single Family": 0.15, "Townhouse": 0.22, "Multi Family": 0.08},
     "streets": ["3rd St", "Tennessee St", "Indiana St", "Minnesota St", "20th St", "22nd St", "Illinois St"],
     "character": "up-and-coming waterfront neighborhood with industrial-chic loft conversions",
     "elem": ["Daniel Webster Elementary", "Bryant Elementary"],
     "middle": ["Hoover Middle School", "Francisco Middle School"],
     "high": ["John O'Connell High School", "City Arts & Tech High School"]},

    {"name": "Glen Park",          "price_mult": 1.15, "w": 50,
     "types": {"Condo": 0.30, "Single Family": 0.48, "Townhouse": 0.15, "Multi Family": 0.07},
     "streets": ["Diamond St", "Bosworth St", "Chenery St", "Elk St", "Surrey St", "Brompton Ave"],
     "character": "tucked-away village with canyon trails and strong neighborhood identity",
     "elem": ["Glen Park Elementary", "Sunnyside Elementary"],
     "middle": ["Aptos Middle School", "James Lick Middle School"],
     "high": ["Abraham Lincoln High School", "Mission High School"]},

    {"name": "West Portal",        "price_mult": 1.22, "w": 50,
     "types": {"Condo": 0.30, "Single Family": 0.52, "Townhouse": 0.12, "Multi Family": 0.06},
     "streets": ["West Portal Ave", "Ulloa St", "Vicente St", "14th Ave", "15th Ave", "Lenox Way", "Claremont Blvd"],
     "character": "charming small-town feel with excellent transit and top-rated schools",
     "elem": ["West Portal Elementary", "Miraloma Elementary"],
     "middle": ["Aptos Middle School", "Hoover Middle School"],
     "high": ["Abraham Lincoln High School", "George Washington High School"]},

    {"name": "Russian Hill",       "price_mult": 1.45, "w": 55,
     "types": {"Condo": 0.55, "Single Family": 0.22, "Townhouse": 0.18, "Multi Family": 0.05},
     "streets": ["Macondray Lane", "Vallejo St", "Broadway", "Green St", "Union St", "Taylor St", "Jones St"],
     "character": "iconic hilltop with Lombard Street, bay views, and charming pedestrian lanes",
     "elem": ["Francisco Elementary", "Garfield Elementary"],
     "middle": ["Francisco Middle School", "Marina Middle School"],
     "high": ["Galileo High School", "George Washington High School"]},

    {"name": "Nob Hill",           "price_mult": 1.40, "w": 65,
     "types": {"Condo": 0.65, "Single Family": 0.12, "Townhouse": 0.15, "Multi Family": 0.08},
     "streets": ["California St", "Sacramento St", "Clay St", "Jones St", "Mason St", "Taylor St", "Powell St"],
     "character": "prestigious hilltop with elegant prewar buildings and cable car access",
     "elem": ["Garfield Elementary", "Spring Valley Elementary"],
     "middle": ["Francisco Middle School", "Marina Middle School"],
     "high": ["Galileo High School", "City Arts & Tech High School"]},

    {"name": "Excelsior",          "price_mult": 0.85, "w": 60,
     "types": {"Condo": 0.25, "Single Family": 0.50, "Townhouse": 0.12, "Multi Family": 0.13},
     "streets": ["Mission St", "Geneva Ave", "Excelsior Ave", "Vienna St", "Naples St", "Persia Ave", "Italy Ave"],
     "character": "diverse, working-class neighborhood with authentic global cuisine",
     "elem": ["Guadalupe Elementary", "Excelsior Elementary", "Junipero Serra Elementary"],
     "middle": ["Aptos Middle School", "Hoover Middle School"],
     "high": ["John O'Connell High School", "Balboa High School"]},

    {"name": "Portola",            "price_mult": 0.80, "w": 50,
     "types": {"Condo": 0.20, "Single Family": 0.55, "Townhouse": 0.12, "Multi Family": 0.13},
     "streets": ["San Bruno Ave", "Felton St", "Burrows St", "Silliman St", "Silver Ave", "Bowdoin St"],
     "character": "quiet residential neighborhood with a growing arts scene and local charm",
     "elem": ["James Denman Middle", "Junipero Serra Elementary", "Longfellow Elementary"],
     "middle": ["James Denman Middle School", "Aptos Middle School"],
     "high": ["Balboa High School", "John O'Connell High School"]},

    {"name": "Bayview",            "price_mult": 0.72, "w": 50,
     "types": {"Condo": 0.22, "Single Family": 0.45, "Townhouse": 0.13, "Multi Family": 0.20},
     "streets": ["3rd St", "Evans Ave", "Cesar Chavez St", "Hudson Ave", "Armstrong Ave", "Innes Ave"],
     "character": "historically rich neighborhood undergoing revitalization with bay views",
     "elem": ["Malcolm X Academy", "Martin Luther King Jr. Elementary"],
     "middle": ["Martin Luther King Jr. Middle School", "Hoover Middle School"],
     "high": ["Thurgood Marshall High School", "Balboa High School"]},

    {"name": "Cole Valley",        "price_mult": 1.20, "w": 40,
     "types": {"Condo": 0.42, "Single Family": 0.35, "Townhouse": 0.15, "Multi Family": 0.08},
     "streets": ["Cole St", "Carl St", "Grattan St", "Shrader St", "Belvedere St", "Frederick St"],
     "character": "small, walkable village between the Haight and Inner Sunset",
     "elem": ["Grattan Elementary", "Harvey Milk Civil Rights Academy"],
     "middle": ["Presidio Middle School", "Aptos Middle School"],
     "high": ["Abraham Lincoln High School", "Galileo High School"]},

    {"name": "Duboce Triangle",    "price_mult": 1.18, "w": 40,
     "types": {"Condo": 0.52, "Single Family": 0.22, "Townhouse": 0.18, "Multi Family": 0.08},
     "streets": ["Sanchez St", "Noe St", "14th St", "15th St", "Duboce Ave", "Waller St", "Henry St"],
     "character": "centrally located, leafy neighborhood popular with young professionals",
     "elem": ["Harvey Milk Civil Rights Academy", "Sanchez Elementary"],
     "middle": ["Everett Middle School", "James Lick Middle School"],
     "high": ["Mission High School", "Abraham Lincoln High School"]},

    {"name": "Twin Peaks",         "price_mult": 1.30, "w": 40,
     "types": {"Condo": 0.38, "Single Family": 0.42, "Townhouse": 0.15, "Multi Family": 0.05},
     "streets": ["Twin Peaks Blvd", "Christmas Tree Point Rd", "Clarendon Ave", "Burnett Ave", "Panorama Dr"],
     "character": "hilltop retreat with 360-degree panoramic views of the entire city",
     "elem": ["Miraloma Elementary", "Glen Park Elementary"],
     "middle": ["Aptos Middle School", "James Lick Middle School"],
     "high": ["Abraham Lincoln High School", "Mission High School"]},

    {"name": "Financial District",  "price_mult": 1.10, "w": 45,
     "types": {"Condo": 0.82, "Single Family": 0.02, "Townhouse": 0.10, "Multi Family": 0.06},
     "streets": ["Market St", "California St", "Montgomery St", "Sansome St", "Battery St", "Spear St"],
     "character": "central urban core with luxury high-rises and immediate access to transit",
     "elem": ["Garfield Elementary", "Spring Valley Elementary"],
     "middle": ["Francisco Middle School", "Marina Middle School"],
     "high": ["Galileo High School", "City Arts & Tech High School"]},

    {"name": "Lower Haight",       "price_mult": 0.95, "w": 45,
     "types": {"Condo": 0.48, "Single Family": 0.22, "Townhouse": 0.18, "Multi Family": 0.12},
     "streets": ["Haight St", "Waller St", "Page St", "Fillmore St", "Webster St", "Steiner St", "Pierce St"],
     "character": "gritty, creative neighborhood with dive bars and vintage shops",
     "elem": ["Buena Vista Elementary", "John Muir Elementary"],
     "middle": ["Everett Middle School", "Presidio Middle School"],
     "high": ["Mission High School", "Galileo High School"]},

    {"name": "Forest Hill",        "price_mult": 1.55, "w": 35,
     "types": {"Condo": 0.15, "Single Family": 0.68, "Townhouse": 0.12, "Multi Family": 0.05},
     "streets": ["Pacheco St", "Magellan Ave", "Castenada Ave", "9th Ave", "Alton Ave", "Dewey Blvd"],
     "character": "serene, wooded enclave with large homes and a private community association",
     "elem": ["West Portal Elementary", "Miraloma Elementary"],
     "middle": ["Aptos Middle School", "Hoover Middle School"],
     "high": ["Abraham Lincoln High School", "George Washington High School"]},
]

# ---------------------------------------------------------------------------
# Architecture styles by property type
# ---------------------------------------------------------------------------

ARCH_STYLES = {
    "Single Family": ["Victorian", "Edwardian", "Craftsman", "Spanish Colonial", "Mid-Century Modern", "Contemporary", "Ranch", "Tudor", "Modern Farmhouse"],
    "Condo":         ["Contemporary", "Modern", "Art Deco", "Mid-Century Modern", "Victorian", "Edwardian", "Loft", "High-Rise"],
    "Townhouse":     ["Contemporary", "Victorian", "Edwardian", "Modern", "Craftsman"],
    "Multi Family":  ["Edwardian", "Victorian", "Contemporary", "Mid-Century Modern", "Spanish Colonial"],
}

# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

VIEWS_BY_NEIGHBORHOOD = {
    "Pacific Heights":   ["Bay View", "Golden Gate View", "City View", "Bay View"],
    "Russian Hill":      ["Bay View", "City View", "Bay View", "Golden Gate View"],
    "Nob Hill":          ["City View", "Bay View", "City View"],
    "Twin Peaks":        ["Panoramic City View", "Twin Peaks View", "Bay View", "Panoramic City View"],
    "Bernal Heights":    ["City View", "Bay View", "Hills View", "City View"],
    "Potrero Hill":      ["City View", "Bay View", "City View"],
    "North Beach":       ["Bay View", "City View"],
    "Marina":            ["Bay View", "Golden Gate View", "Bay View"],
    "SOMA":              ["Bay View", "City View"],
    "Financial District": ["Bay View", "City View", "Bay View"],
    "Forest Hill":       ["City View", "Park View", "Hills View"],
    "Bayview":           ["Bay View", "City View"],
}
DEFAULT_VIEWS = ["None", "None", "None", "Garden View", "Park View", "City View"]

# ---------------------------------------------------------------------------
# Listing description templates
# ---------------------------------------------------------------------------

def make_description(listing):
    pt = listing["property_type"]
    arch = listing["arch_style"]
    beds = listing["total_bedrooms"]
    baths = listing["total_bathrooms"]
    sqft = listing["total_square_feet"]
    price = listing["list_price"]
    nbhd = listing["neighborhood_name"]
    view = listing["views"]
    nbhd_char = listing["_nbhd_char"]
    street = listing["_street"]

    has_view = view != "None"
    view_phrase = f" soaking in {view.lower()}s" if has_view else ""
    view_sentence = f" Enjoy breathtaking {view.lower()}s from the living areas and primary suite." if has_view else ""

    parking_options = ["1-car garage", "2-car garage", "tandem garage", "deeded parking", "1-car parking", "EV-ready garage"]
    outdoor_options = ["a private rear garden", "a sunny deck", "a rooftop terrace", "a shared roof deck with panoramic views",
                       "a landscaped patio", "a charming Juliet balcony", "an expansive yard", "a wrap-around deck"]
    update_options = ["chef's kitchen with Carrara marble countertops and stainless appliances",
                      "fully remodeled kitchen with custom cabinetry and quartz countertops",
                      "updated kitchen with gas range and subway tile backsplash",
                      "gourmet kitchen with waterfall island and wine refrigerator",
                      "designer kitchen with integrated appliances and pantry",
                      "bright eat-in kitchen with breakfast nook and garden window"]
    laundry_options = ["in-unit washer/dryer", "washer/dryer hookups", "dedicated laundry room", "in-unit laundry closet"]
    light_options = ["sun-drenched", "light-filled", "bright and airy", "sun-soaked", "naturally lit", "luminous"]

    parking = random.choice(parking_options)
    outdoor = random.choice(outdoor_options)
    kitchen = random.choice(update_options)
    laundry = random.choice(laundry_options)
    light = random.choice(light_options)

    TEMPLATES = {
        "Single Family": [
            f"Welcome to this stunning {arch} home nestled in the heart of {nbhd}. "
            f"This {light} {beds}-bedroom, {baths}-bath residence offers {sqft:,} sq ft of thoughtfully designed living space{view_phrase}. "
            f"The {kitchen} anchors the main floor, opening to {outdoor}. "
            f"Original architectural details — including {random.choice(['period moldings and hardwood floors', 'bay windows and wainscoting', 'coffered ceilings and built-ins', 'original fireplace mantel and picture rails'])} — "
            f"have been lovingly preserved. {light.capitalize()} primary suite with spa bath. {laundry.capitalize()} and {parking} complete the package.{view_sentence} "
            f"Steps from {nbhd}'s {random.choice(['best cafés and parks', 'acclaimed restaurants and transit', 'farmers market and boutiques', 'top-rated schools and green spaces'])}.",

            f"Exceptional {arch} {beds}BD/{baths}BA in coveted {nbhd}. "
            f"Spanning {sqft:,} sq ft, this {light} home features {kitchen}, "
            f"gleaming hardwood floors, and {outdoor}. "
            f"The spacious primary suite offers {random.choice(['an ensuite bath with soaking tub', 'a walk-in closet and spa-inspired bath', 'a private deck and ensuite bath'])}. "
            f"Additional highlights: {laundry}, {parking}, formal dining room, and {random.choice(['a cozy wood-burning fireplace', 'a gas fireplace for cool SF evenings', 'a statement fireplace in the living room'])}. "
            f"Situated on a {random.choice(['quiet tree-lined block', 'sunny corner lot', 'wide flat lot', 'premium block'])} in {nbhd} — {nbhd_char}.{view_sentence}",

            f"Rare opportunity to own a meticulously restored {arch} in {nbhd}. "
            f"This {beds}-bed, {baths}-bath home ({sqft:,} sq ft) balances period character with modern updates. "
            f"The {kitchen} leads to {outdoor}, ideal for entertaining. "
            f"Upstairs, the {light} primary bedroom features {random.choice(['vaulted ceilings', 'period windows', 'an ensuite bath and walk-in closet'])}. "
            f"Original {random.choice(['fir floors', 'hardwood floors', 'oak floors'])} throughout. {parking.capitalize()} and {laundry}.{view_sentence} "
            f"{nbhd} is celebrated for {nbhd_char} — this home delivers it all.",
        ],
        "Condo": [
            f"Sophisticated {beds}BD/{baths}BA {arch.lower()} condo in the heart of {nbhd}. "
            f"This {light} {sqft:,} sq ft residence features {kitchen}, "
            f"open-concept living/dining, and floor-to-ceiling windows. "
            f"The primary suite offers {random.choice(['a spa-inspired bath with dual vanity', 'a walk-in closet and ensuite bath', 'an ensuite bath with heated floors'])}. "
            f"{laundry.capitalize()}, {parking}, and {outdoor} round out this exceptional offering.{view_sentence} "
            f"Walk to {nbhd}'s {random.choice(['best dining and boutiques', 'Michelin-starred restaurants and nightlife', 'parks, cafés, and transit'])}.",

            f"Welcome to unit #{random.randint(2,24)}{random.choice(['A','B','C','D'])} — a {light} {beds}BD/{baths}BA residence in one of {nbhd}'s most desirable buildings. "
            f"At {sqft:,} sq ft, this {arch.lower()} condo offers {kitchen}, "
            f"custom built-ins, and {random.choice(['wide-plank white oak floors', 'polished concrete floors', 'engineered hardwood floors', 'heated tile floors'])}. "
            f"Amenities include {random.choice(['a rooftop lounge', 'a fitness center and concierge', 'a rooftop terrace and club room', 'bike storage and package room'])}. "
            f"{laundry.capitalize()}, {parking}.{view_sentence}",

            f"Modern living in the vibrant {nbhd} neighborhood. "
            f"This sleek {beds}BD/{baths}BA {arch.lower()} condo ({sqft:,} sq ft) features {kitchen}, "
            f"Bosch appliances, and {outdoor}. "
            f"{light.capitalize()} throughout with {random.choice(['double-pane windows for sound insulation', 'oversized windows flooding rooms with natural light', 'skylights in the primary bath'])}. "
            f"Building features {random.choice(['secure entry, elevator, and bike room', 'gym, rooftop, and storage', 'concierge, fitness center, and guest suite'])}. "
            f"{laundry.capitalize()} and {parking} included.{view_sentence}",
        ],
        "Townhouse": [
            f"Stunning {beds}BD/{baths}BA {arch} townhouse in {nbhd} — the best of both worlds: the space of a home with the ease of condo living. "
            f"Spanning {sqft:,} sq ft across {random.choice(['three', 'four'])} levels, this {light} residence features {kitchen}, "
            f"a private rooftop deck, and {parking}. "
            f"Open living/dining on the main level flows to {outdoor}. "
            f"The primary suite occupies its own floor with {random.choice(['a spa bath, dual vanities, and walk-in closet', 'an ensuite bath and private terrace', 'a soaking tub and dual walk-in closets'])}. "
            f"{laundry.capitalize()} on bedroom level.{view_sentence}",

            f"Exceptional end-unit {arch} townhouse in the heart of {nbhd}. "
            f"{beds} bedrooms, {baths} baths, {sqft:,} sq ft. "
            f"Designed for modern living with {kitchen}, "
            f"open-plan great room, and {outdoor}. "
            f"Generous primary suite with {random.choice(['spa bath and walk-in closet', 'ensuite bath and custom built-ins', 'dual vanity bath and closet'])}. "
            f"{parking.capitalize()}, {laundry}, smart home features throughout.{view_sentence} "
            f"Walk to {nbhd}'s acclaimed {random.choice(['restaurants, parks, and Muni', 'boutiques, cafés, and transit', 'farmers market and green spaces'])}.",
        ],
        "Multi Family": [
            f"Prime {nbhd} {random.choice(['duplex', 'triplex', 'two-unit building'])} — a rare turnkey investment or owner-occupy opportunity. "
            f"This {arch} building offers {beds} total bedrooms, {baths} baths, and {sqft:,} sq ft of living space. "
            f"Each unit features {random.choice(['updated kitchens, hardwood floors, and high ceilings', 'remodeled kitchens and bathrooms, and in-unit laundry', 'period details, modern kitchens, and separate entrances'])}. "
            f"Building includes {parking} and {outdoor}. "
            f"Strong rental history in one of SF's most desirable neighborhoods.{view_sentence}",

            f"Investor special or perfect for multigenerational living — a fully updated {arch} {random.choice(['duplex', 'triplex'])} in sought-after {nbhd}. "
            f"{beds} bedrooms and {baths} baths across {sqft:,} sq ft. "
            f"Both units feature {random.choice(['in-unit laundry, updated kitchens, and separate meters', 'modern kitchens, hardwood floors, and separate entrances', 'spacious living areas, updated baths, and abundant storage'])}. "
            f"{parking.capitalize()}, separate utilities, and a {random.choice(['large shared yard', 'shared rear garden', 'generous garage with storage'])}. "
            f"Trophy {nbhd} address with strong cap rate potential.{view_sentence}",
        ],
    }

    templates = TEMPLATES.get(pt, TEMPLATES["Condo"])
    return random.choice(templates).strip()

# ---------------------------------------------------------------------------
# Price generation
# ---------------------------------------------------------------------------

PRICE_RANGES = {
    "Single Family": (1_100_000, 3_000_000),
    "Condo":         (  500_000, 1_500_000),
    "Townhouse":     (  800_000, 2_000_000),
    "Multi Family":  (  900_000, 3_000_000),
}

def gen_price(prop_type, price_mult):
    lo, hi = PRICE_RANGES[prop_type]
    lo = max(500_000, int(lo * price_mult))
    hi = min(3_000_000, int(hi * price_mult))
    raw = random.randint(lo, hi)
    # round to nearest 25k for realism
    return round(raw / 25_000) * 25_000

# ---------------------------------------------------------------------------
# Beds / baths / sqft
# ---------------------------------------------------------------------------

UNIT_SPECS = {
    "Single Family": {
        "beds": [2, 3, 3, 3, 4, 4, 5],
        "bath_add": [0, 0.5, 1, 1],
        "sqft_base": (1_200, 3_800),
    },
    "Condo": {
        "beds": [0, 1, 1, 2, 2, 2, 3],
        "bath_add": [0, 0.5, 1],
        "sqft_base": (450, 1_800),
    },
    "Townhouse": {
        "beds": [2, 2, 3, 3, 4],
        "bath_add": [0.5, 1, 1, 1.5],
        "sqft_base": (900, 2_600),
    },
    "Multi Family": {
        "beds": [3, 4, 4, 5, 6],
        "bath_add": [0, 0.5, 1, 1],
        "sqft_base": (1_800, 5_000),
    },
}

def gen_unit(prop_type):
    spec = UNIT_SPECS[prop_type]
    beds = random.choice(spec["beds"])
    baths = max(1.0, beds + random.choice(spec["bath_add"]))
    baths = round(baths * 2) / 2  # nearest 0.5
    sqft_lo, sqft_hi = spec["sqft_base"]
    sqft = random.randint(sqft_lo, sqft_hi)
    sqft = round(sqft / 50) * 50  # round to 50 sqft
    return beds, baths, sqft

# ---------------------------------------------------------------------------
# MLS number generator
# ---------------------------------------------------------------------------

def gen_mls():
    return f"SF{random.randint(100000, 999999)}"

# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def pick_property_type(type_weights):
    types = list(type_weights.keys())
    weights = list(type_weights.values())
    return random.choices(types, weights=weights, k=1)[0]

def pick_view(nbhd_name):
    pool = VIEWS_BY_NEIGHBORHOOD.get(nbhd_name, DEFAULT_VIEWS)
    return random.choice(pool)

def build_address(street):
    number = random.randint(10, 3999)
    unit_suffix = ""
    if random.random() < 0.35:
        unit_suffix = f" #{random.randint(1, 30)}{random.choice(['', 'A', 'B', 'C'])}"
    return f"{number} {street}{unit_suffix}, San Francisco, CA"

def generate_listings(target=2000):
    listings = []
    seen_mls = set()

    # Build weighted neighborhood pool
    total_weight = sum(n["w"] for n in NEIGHBORHOODS)
    neighborhood_pool = []
    for n in NEIGHBORHOODS:
        count = max(1, round(target * n["w"] / total_weight))
        neighborhood_pool.extend([n] * count)
    random.shuffle(neighborhood_pool)

    # Pad or trim to exactly target
    while len(neighborhood_pool) < target:
        neighborhood_pool.append(random.choice(NEIGHBORHOODS))
    neighborhood_pool = neighborhood_pool[:target]

    statuses = ["Active"] * 85 + ["Coming Soon"] * 15

    for i, nbhd in enumerate(neighborhood_pool):
        # Unique MLS
        while True:
            mls = gen_mls()
            if mls not in seen_mls:
                seen_mls.add(mls)
                break

        prop_type = pick_property_type(nbhd["types"])
        arch_style = random.choice(ARCH_STYLES[prop_type])
        beds, baths, sqft = gen_unit(prop_type)
        price = gen_price(prop_type, nbhd["price_mult"])
        street = random.choice(nbhd["streets"])
        address = build_address(street)
        view = pick_view(nbhd["name"])
        status = random.choice(statuses)

        elem   = random.choice(nbhd["elem"])
        middle = random.choice(nbhd["middle"])
        high   = random.choice(nbhd["high"])

        listing = {
            "mls_listing_number":   mls,
            "address":              address,
            "list_price":           price,
            "total_bedrooms":       beds,
            "total_bathrooms":      baths,
            "total_square_feet":    sqft,
            "property_type":        prop_type,
            "neighborhood_name":    nbhd["name"],
            "school_district_name": "San Francisco Unified School District",
            "elementary_school_name": elem,
            "middle_school_name":   middle,
            "high_school_name":     high,
            "views":                view,
            # internal fields used only for description generation
            "_nbhd_char":           nbhd["character"],
            "_street":              street,
            "arch_style":           arch_style,
        }
        listing["listing_description"] = make_description(listing)

        # Remove internal fields before saving
        listing["architecture_style"] = arch_style
        clean = {k: v for k, v in listing.items() if not k.startswith("_") and k != "arch_style"}
        listings.append(clean)

    return listings

def generate_photos(listings):
    photos = []
    for listing in listings:
        mls = listing["mls_listing_number"]
        n_photos = random.randint(5, 12)
        for j in range(1, n_photos + 1):
            photos.append({
                "mls_listing_number": mls,
                "main_photo": j == 1,
                "photo": f"{mls}_{j:02d}.jpg",
            })
    return photos

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    print("Generating 2000 SF listings...")
    listings = generate_listings(2000)

    listings_path = out_dir / "listings.csv"
    listing_cols = [
        "mls_listing_number", "address", "list_price", "total_bedrooms",
        "total_bathrooms", "total_square_feet", "property_type", "architecture_style",
        "neighborhood_name", "school_district_name", "elementary_school_name",
        "middle_school_name", "high_school_name", "listing_description", "views",
    ]
    with open(listings_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=listing_cols)
        writer.writeheader()
        writer.writerows(listings)
    print(f"  Saved {len(listings)} listings → {listings_path}")

    print("Generating listing photos...")
    photos = generate_photos(listings)

    photos_path = out_dir / "listing_photos.csv"
    photo_cols = ["mls_listing_number", "main_photo", "photo"]
    with open(photos_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=photo_cols)
        writer.writeheader()
        writer.writerows(photos)
    print(f"  Saved {len(photos)} photo rows → {photos_path}")

    # Distribution summary
    from collections import Counter
    types = Counter(l["property_type"] for l in listings)
    nbhds = Counter(l["neighborhood_name"] for l in listings)
    print("\nProperty type distribution:")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print(f"\nNeighborhoods covered: {len(nbhds)}")
    print(f"  Largest: {nbhds.most_common(3)}")
    print(f"  Smallest: {nbhds.most_common()[-3:]}")
    prices = [l["list_price"] for l in listings]
    print(f"\nPrice range: ${min(prices):,} – ${max(prices):,}")
    print(f"  Median: ${sorted(prices)[len(prices)//2]:,}")
    print("\nDone.")