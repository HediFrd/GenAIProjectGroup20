# scripts/create_instruction_bank.py
import json

def create_fine_grained_instruction_bank():
    """Create a bank of 30 fine-grained makeup instructions"""
    
    instruction_bank = {
        "eyes": [
            "Add subtle brown eyeshadow",
            "Apply dramatic smokey eye makeup",
            "Create a natural eye look with light shimmer",
            "Add winged eyeliner to both eyes",
            "Apply purple eyeshadow to the upper eyelids",
            "Intensify the eye makeup with darker tones",
            "Add golden glitter to the inner corners",
            "Create a cut crease with bold colors"
        ],
        "lips": [
            "Apply nude lipstick",
            "Make lips red and glossy",
            "Add dark burgundy lip color",
            "Apply pink lip gloss",
            "Create ombre lip effect",
            "Add matte brown lipstick",
            "Make lips more plump and defined",
            "Apply coral lipstick"
        ],
        "skin": [
            "Add natural-looking foundation",
            "Apply light contouring to cheekbones",
            "Add subtle blush to the cheeks",
            "Create a dewy skin finish",
            "Apply bronzer for a sun-kissed look",
            "Add highlighter to high points",
            "Create matte skin finish",
            "Apply light pink blush"
        ],
        "brows": [
            "Define eyebrows more prominently",
            "Create natural-looking brows",
            "Darken the eyebrows slightly",
            "Add arch to the eyebrows"
        ],
        "overall": [
            "Apply natural everyday makeup",
            "Create glamorous evening makeup",
            "Add festival-style makeup with colors",
            "Apply minimal no-makeup makeup look",
            "Create editorial high-fashion makeup",
            "Add vintage-inspired makeup"
        ]
    }
    
    # Save to JSON
    output_path = "data/instruction_bank.json"
    with open(output_path, 'w') as f:
        json.dump(instruction_bank, f, indent=2)
    
    print(f"Created instruction bank with {sum(len(v) for v in instruction_bank.values())} instructions")
    print(f"Saved to {output_path}")
    
    return instruction_bank

if __name__ == "__main__":
    create_fine_grained_instruction_bank()
