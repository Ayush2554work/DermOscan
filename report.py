import os  # <-- ADDED THIS IMPORT
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import customtkinter
from PIL import Image

def generate_report_pdf(report_data):
    """Generates and saves a PDF report from the report_data dictionary."""
    
    file_path = customtkinter.filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        initialfile=f"DermOscan_Report_{report_data['report_id']}.pdf"
    )
    
    if not file_path:
        return False # User cancelled

    try:
        w, h = A4
        c = canvas.Canvas(file_path, pagesize=A4)

        # --- [NEW] Add Logo ---
        logo_path = 'logo.png'  # As requested, it will look for logo.png
        if os.path.exists(logo_path):
            try:
                # Draw logo in top-left corner
                # (1 inch from left, 1.25 inches from top)
                c.drawImage(logo_path, 1 * inch, h - 1.25 * inch, 
                            width=1*inch, height=1*inch, 
                            preserveAspectRatio=True, mask='auto')
            except Exception as e:
                print(f"Error drawing logo: {e}")
        
        # --- Title ---
        c.setFont('Helvetica-Bold', 18)
        c.drawCentredString(w / 2.0, h - 1.0 * inch, "DermOscan - AI Skin Analysis Report")
        
        # --- Report ID ---
        c.setFont('Helvetica', 12)
        c.drawString(1 * inch, h - 1.5 * inch, f"Report ID: {report_data['report_id']}")
        
        # --- Spacer ---
        c.line(1 * inch, h - 1.7 * inch, w - 1 * inch, h - 1.7 * inch)
        
        # --- Summary Section ---
        c.setFont('Helvetica-Bold', 14)
        c.drawString(1 * inch, h - 2.1 * inch, "Prediction Summary")
        
        c.setFont('Helvetica', 12)
        c.drawString(1.2 * inch, h - 2.4 * inch, f"Status: {report_data['status']}")
        c.drawString(1.2 * inch, h - 2.6 * inch, f"Prediction: {report_data['prediction']}")
        c.drawString(1.2 * inch, h - 2.8 * inch, f"Model Confidence: {report_data['confidence']}")

        # --- Analyzed Image ---
        img_path = report_data['image_path']
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            img_display_width = 2.5 * inch
            img_display_height = img_display_width * aspect
            
            c.drawImage(
                img_path,
                w - 1 * inch - img_display_width, 
                h - 2.1 * inch - img_display_height,
                width=img_display_width,
                height=img_display_height,
                mask='auto'
            )

        # --- Details Section ---
        c.setFont('Helvetica-Bold', 14)
        current_y = h - 3.4 * inch
        if img_display_height > (1.3 * inch): # Move details down if image is tall
             current_y = h - 2.3 * inch - img_display_height
                
        c.drawString(1 * inch, current_y, "Key Factors & Analysis")
        current_y -= 0.3 * inch

        c.setFont('Helvetica', 11)
        for key, value in report_data['factors'].items():
            c.drawString(1.2 * inch, current_y, f"•  {key}: {value}")
            current_y -= 0.25 * inch

        # --- Recommendation Section ---
        current_y -= 0.3 * inch
        c.setFont('Helvetica-Bold', 14)
        c.drawString(1 * inch, current_y, "Recommendation")
        current_y -= 0.3 * inch
        
        # --- Text Wrapping for Recommendation ---
        c.setFont('Helvetica-Oblique', 11)
        text_object = c.beginText(1.2 * inch, current_y)
        text_object.setFont('Helvetica-Oblique', 11)
        
        wrap_width = w - 2.4 * inch
        lines = []
        for line in report_data['recommendation'].split('\n'):
            words = line.split()
            current_line = ""
            for word in words:
                if c.stringWidth(current_line + " " + word, 'Helvetica-Oblique', 11) < wrap_width:
                    current_line += " " + word
                else:
                    lines.append(current_line.strip())
                    current_line = word
            lines.append(current_line.strip())
            
        for line in lines:
            text_object.textLine(line)
        
        text_object_y = text_object.getY() # Get position after drawing text
        c.drawText(text_object)
        
        # --- Footer ---
        # Adjust footer position based on text, but set a minimum
        footer_y = min(current_y - (len(lines) * 0.2 * inch) - 0.5 * inch, 2.0 * inch)
        
        c.setFont('Helvetica', 9)
        c.line(1 * inch, footer_y, w - 1 * inch, footer_y)
        c.drawCentredString(w / 2.0, footer_y - 0.2 * inch, 
            "Disclaimer: This is an AI-generated report for informational purposes only. "
            "It is not a medical diagnosis.")
        c.drawCentredString(w / 2.0, footer_y - 0.4 * inch, 
            "Always consult a qualified dermatologist for any health concerns.")
        
        # --- [NEW] Add Name to Footer ---
        c.setFont('Helvetica-Bold', 9) # Make it bold
        c.drawCentredString(w / 2.0, footer_y - 0.6 * inch, "[ Application Developed by AYUSH ]")
        
        # Save the PDF
        c.showPage()
        c.save()
        return True
    
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False