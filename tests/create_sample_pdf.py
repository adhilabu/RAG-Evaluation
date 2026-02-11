"""Generate a sample PDF for testing."""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from pathlib import Path


def create_sample_pdf(output_path: str, num_pages: int = 20):
    """
    Create a sample multi-page PDF for testing.
    
    Args:
        output_path: Path to save the PDF
        num_pages: Number of pages to generate
    """
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    for page_num in range(1, num_pages + 1):
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1*inch, height - 1*inch, f"Sample Document - Page {page_num}")
        
        # Content
        c.setFont("Helvetica", 12)
        y_position = height - 1.5*inch
        
        # Add some sample content
        content = [
            f"This is page {page_num} of a sample document created for testing purposes.",
            "",
            "Section 1: Introduction",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor",
            "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis",
            "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
            "",
            "Section 2: Key Findings",
            f"- Finding {page_num}.1: Important discovery on page {page_num}",
            f"- Finding {page_num}.2: Significant result related to item {page_num}",
            f"- Finding {page_num}.3: Critical observation for section {page_num}",
            "",
            "Section 3: Details",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore",
            "eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt",
            "in culpa qui officia deserunt mollit anim id est laborum.",
            "",
            f"The page number is {page_num} and this document contains important information",
            "that demonstrates the map-reduce summarization and RAG storage capabilities.",
            "",
            "Additional content to make the page more substantial and test chunking:",
            "- Point A: Description of point A with relevant details",
            "- Point B: Description of point B with supporting evidence",
            "- Point C: Description of point C with contextual information",
        ]
        
        for line in content:
            c.drawString(1*inch, y_position, line)
            y_position -= 0.25*inch
            
            if y_position < 1*inch:  # Don't go too far down
                break
        
        # Page number at bottom
        c.setFont("Helvetica", 10)
        c.drawString(width/2 - 0.5*inch, 0.5*inch, f"Page {page_num} of {num_pages}")
        
        c.showPage()
    
    c.save()
    print(f"Created sample PDF: {output_path} ({num_pages} pages)")


if __name__ == "__main__":
    import sys
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    num_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    
    output_path = output_dir / f"sample_{num_pages}page.pdf"
    create_sample_pdf(str(output_path), num_pages)
