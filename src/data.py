import sys
import os
import time
import shutil
import torch
from tqdm import tqdm
from urllib.request import urlretrieve
from Bio.Seq import Seq
from src.utils import create_dir, shift_sequence


class SequenceReadingsDataset(torch.utils.data.Dataset):
    def __init__(self, test_type: str) -> None:
        """ Creates PyTorch dataset from a FASTA file and prepares it for the pipeline

        Args:
            fasta_file (str): Path to the FASTA file.
        """
        super().__init__()

        self.sequence_id = []
        self.aa_sequence = []
        self.contains_stop = []
        self.label_frame = []
        self.dna_sequence = []
        self.label_species = []

        self._download_testsets()

        if test_type == "frame":
            fasta_file = "./data/refseq/refseq_ds_all_off-frames_fb_DNA_test.fasta"
        elif test_type == "taxonomic":
            fasta_file = "./data/uniprot/uniprot_swiss-prot_vbh_p100d_w_test.fasta"
        elif test_type == "inORF":
            fasta_file = "./data/inORF/inORF_unique.fasta"
        elif test_type == "SRR":
            fasta_file = "./data/srr/SRR2940986_filtered.fasta"
        else:
            fasta_file = test_type

        with open(fasta_file) as file:
            for line in tqdm(file):
                line = line.strip()
                if line.startswith(">"):
                    # Based on the file type handle the header line differently
                    if test_type == "frame":
                        self.sequence_id.append(line.split("|")[0])
                        self.label_frame.append(int(line[-1]))
                    elif test_type == "taxonomic":
                        self.sequence_id.append(line.split("|")[0])
                        self.label_species.append(int(line[-1]))
                    else:
                        self.sequence_id.append(line)
                        if any(label in line for label in ("viral", "virus")):
                            self.label_species.append(0)
                        if "bacterial" in line:
                            self.label_species.append(1)
                        if any(label in line for label in ("human", "pig")):
                            self.label_species.append(2)
                else:
                    # Cut sequences into a length of 3
                    if len(line) % 3 != 0:
                        off_set = len(line) - len(line) % 3
                        line = line[:off_set]
                    if not test_type == "taxonomic":
                        # Have a look at the amino acid and check if it has a STOP codon in it
                        for frame in range(0, 6):
                            aa_seq = str(Seq(shift_sequence(line, frame)).translate())
                            # If no STOP codon is in it replace the original sequence
                            if "*" not in aa_seq:
                                line = shift_sequence(line, frame)
                                break                 
                        self.dna_sequence.append(line)
                        line = Seq(line).translate()
                    # Translate DNA and prepare string for the explorer

                    self.aa_sequence.append(" ".join(line))
                    # Tag if the translation got a STOP codon in it
                    if "*" in line:
                        self.contains_stop.append(True)
                    else:
                        self.contains_stop.append(False)

    def _download_testsets(self):
        """ Downloads the pretrained classification models for a certain model type.
        """
        download_urls = {
            "frame": "https://zenodo.org/record/4306248/files/refseq.tar.gz",
            "taxonomic": "https://zenodo.org/record/4306240/files/uniprot.tar.gz",
            "SRR": "https://redmine.f4.htw-berlin.de/owncloud/index.php/s/NoXtz6ezSZHPB6T/download",
            "inORF": "https://redmine.f4.htw-berlin.de/owncloud/index.php/s/REkM3Zi5K8n9QW2/download"
        }
        
        def reporthook(count: int, block_size: int, total_size: int) -> None:
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write("\r %d%% | %d MB | %d KB/s" % (percent, progress_size / (1024 * 1024), speed))
            sys.stdout.flush()

        for key in download_urls:
            if key == "frame" and os.path.isfile("./data/refseq/refseq_ds_all_off-frames_fb_DNA_test.fasta"): continue
            if key == "taxonomic" and os.path.isfile("./data/uniprot/uniprot_swiss-prot_vbh_p100d_w_test.fasta"): continue
            if key == "SRR" and os.path.isfile("./data/srr/SRR2940986_filtered.fasta"): continue
            if key == "inORF" and os.path.isfile("./data/inORF/inORF_unique.fasta"): continue
            # Download
            dir_path = "./data"
            create_dir(dir_path)
            file_name = download_urls[key].split("/")[-1]
            file_path = os.path.join(dir_path, file_name)
            urlretrieve(download_urls[key], filename=file_path, reporthook=reporthook)
            print(" - {} successfully downloaded".format(file_name))
            # Unzip
            shutil.unpack_archive(file_path, extract_dir=dir_path, format="gztar")
            # Remove downloaded archive
            os.remove(file_path)

    def __getitem__(self, index):
        return self.aa_sequence[index]

    def __len__(self):
        return len(self.sequence_id)


class Frame_Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        super().__init__()
        self.corrected_seq = sequences

    def __getitem__(self, index):
        return self.corrected_seq[index]

    def __len__(self):
        return len(self.corrected_seq)
