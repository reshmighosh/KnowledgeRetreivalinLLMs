from pypdf import PdfMerger
import os


root_dir = "/work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/rome/results/llama_inst_context/causal_trace/pdfs"

pdfs = os.listdir(root_dir)

pdfs = [os.path.join(root_dir, p) for p in pdfs]

merger = PdfMerger()

for pdf in pdfs:
    merger.append(pdf)

merger.write("llama_inst_context.pdf")
merger.close()