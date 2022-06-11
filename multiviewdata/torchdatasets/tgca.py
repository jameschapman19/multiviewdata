import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

tumor_list = [
    "ACC",
    "BLCA",
    "BRCA",
    "CESC",
    "CHOL",
    "COAD",
    "COADREAD",
    "DLBC",
    "ESCA",
    "FPPP",
    "GBM",
    "GBMLGG",
    "HNSC",
    "KICH",
    "KIPAN",
    "KIRC",
    "KIRP",
    "LAML",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "MESO",
    "OV",
    "PAAD",
    "PCPG",
    "PRAD",
    "READ",
    "SARC",
    "SKCM",
    "STAD",
    "STES",
    "TGCT",
    "THCA",
    "THYM",
    "UCEC",
    "UCS",
    "UVM",
]


def process_file(file):
    tmp = pd.read_csv(file, sep="\t", error_bad_lines=False)
    tmp.drop_duplicates(inplace=True)
    tmp.columns = [list(tmp)[0]] + [f[:15] for f in list(tmp)[1:]]
    tmp = tmp.T.reset_index()
    tmp.columns = tmp.iloc[0, 0:]
    tmp = tmp.iloc[1:, :].reset_index(drop=True)
    return tmp.loc[:, ~tmp.columns.duplicated(keep="first")]


def clinical(tumor_list, root="", folder="/gdac.broadinstitute.org_"):
    """
    This function downloads the clinical data from the Broad Institute and processes it. It returns a dataframe with the
    clinical data.

    :param tumor_list: list of tumor types
    :param root: path to the directory where the data is stored
    :param folder:
    :return:
    """
    for tumor in tumor_list:
        file = root + folder + f"{tumor}.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz"
        if os.path.exists(file):
            tmp = process_file(file)
            tmp = tmp.rename(columns={tmp.columns[0]: "Hybridization REF"})
            if tumor == "ACC":
                label_df = tmp
            else:
                label_df = pd.concat(
                    [label_df, tmp.drop("././@LongLink", axis=1, errors="ignore")],
                    axis=0,
                    ignore_index=True,
                )
    label_df = label_df.sort_values(by="Hybridization REF").reset_index(drop=True)
    label_df = (
        label_df[label_df["Hybridization REF"].apply(lambda x: "tcga" in x)]
            .drop_duplicates(subset=["Hybridization REF"], keep="last")
            .reset_index(drop=True)
    )
    label_df["1yr-mortality"] = -1.0
    label_df.loc[label_df["days_to_last_followup"].astype(float) >= 365, "1yr-mortality"] = 0.0
    label_df.loc[label_df["days_to_death"].astype(float) > 365, "1yr-mortality"] = 0.0
    label_df.loc[label_df["days_to_death"].astype(float) <= 365, "1yr-mortality"] = 1.0

    label_df["3yr-mortality"] = -1.0
    label_df.loc[label_df["days_to_last_followup"].astype(float) >= 3 * 365, "3yr-mortality"] = 0.0
    label_df.loc[label_df["days_to_death"].astype(float) > 3 * 365, "3yr-mortality"] = 0.0
    label_df.loc[label_df["days_to_death"].astype(float) <= 3 * 365, "3yr-mortality"] = 1.0

    label_df["5yr-mortality"] = -1.0
    label_df.loc[label_df["days_to_last_followup"].astype(float) >= 5 * 365, "5yr-mortality"] = 0.0
    label_df.loc[label_df["days_to_death"].astype(float) > 5 * 365, "5yr-mortality"] = 0.0
    label_df.loc[label_df["days_to_death"].astype(float) <= 5 * 365, "5yr-mortality"] = 1.0
    label_df.to_csv(root + "/FINAL/clinical_label.csv", index=False)


def rppa(tumor_list, root="", folder="/gdac.broadinstitute.org_"):
    """
    This function downloads the rppa data from the Broad Institute and processes it. It returns a dataframe with the
    rppa data.

    :param tumor_list:
    :param root:
    :param folder:
    :return:
    """
    for tumor in tumor_list:
        file = (
                root
                + folder
                + f"{tumor}.RPPA_AnnotateWithGene.Level_3.2016012800.0.0/{tumor}.rppa.txt"
        )
        if os.path.exists(file):
            tmp = process_file(file)
            if tumor == "ACC":
                rppa_df = tmp
            else:
                rppa_df = pd.concat([rppa_df, tmp], axis=0)
    rppa_df = rppa_df.rename(columns={"Composite.Element.REF": "Hybridization REF"})
    rppa_df["Hybridization REF"] = rppa_df["Hybridization REF"].apply(
        lambda x: x.lower()[:-3]
    )
    rppa_df = rppa_df.drop_duplicates(subset=["Hybridization REF"])
    tmp_list = np.asarray(list(rppa_df))
    rppa_df = rppa_df[tmp_list[rppa_df.isna().sum(axis=0) == 0]]
    rppa_df.to_csv(root + "/FINAL/RPPA.csv", index=False)


def mrnaseq(tumor_list, root="", folder="/gdac.broadinstitute.org_"):
    """

    Parameters
    ----------
    tumor_list
    root
    folder

    Returns
    -------

    """
    for tumor in tumor_list:
        file = (
                root
                + folder
                + f"{tumor}.mRNAseq_Preprocess.Level_3.2016012800.0.0/{tumor}.uncv2.mRNAseq_RSEM_normalized_log2.txt"
        )
        if os.path.exists(file):
            tmp = process_file(file)
            if tumor == "ACC":
                mrna_df = tmp
            else:
                mrna_df = pd.concat([mrna_df, tmp], axis=0)
    mrna_df = mrna_df.rename(columns={"gene": "Hybridization REF"})
    mrna_df["Hybridization REF"] = mrna_df["Hybridization REF"].apply(
        lambda x: x.lower()[:-3]
    )
    mrna_df = mrna_df.drop_duplicates(subset=["Hybridization REF"])
    tmp_list = np.asarray(list(mrna_df))
    mrna_df = mrna_df[tmp_list[mrna_df.isna().sum(axis=0) == 0]]
    mrna_df.to_csv(root + "/FINAL/mRNAseq_RSEM.csv", index=False)


def mirnaseq(tumor_list, root="", folder="/gdac.broadinstitute.org_"):
    """

    Args:
        tumor_list:
        root:
        folder:

    Returns:

    """
    for tumor in tumor_list:
        file = (
                root
                + folder
                + f"{tumor}.miRseq_Preprocess.Level_3.2016012800.0.0/{tumor}.miRseq_RPKM_log2.txt"
        )
        if os.path.exists(file):
            tmp = process_file(file)
            if tumor == "ACC":
                mirna_df = tmp
            else:
                mirna_df = pd.concat([mirna_df, tmp], axis=0)
    mirna_df = mirna_df.rename(columns={"gene": "Hybridization REF"})
    mirna_df["Hybridization REF"] = mirna_df["Hybridization REF"].apply(
        lambda x: x.lower()[:-3]
    )
    mirna_df = mirna_df.drop_duplicates(subset=["Hybridization REF"])
    tmp_list = np.asarray(list(mirna_df))
    mirna_df = mirna_df[tmp_list[mirna_df.isna().sum(axis=0) == 0]]
    mirna_df.to_csv(root + "/FINAL/miRNAseq_RPKM_log2.csv", index=False)


def methylation(tumor_list, root="", folder="/gdac.broadinstitute.org_"):
    for tumor in tumor_list:
        file = (
                root
                + folder
                + f"{tumor}.Methylation_Preprocess.Level_3.2016012800.0.0/{tumor}.meth.by_mean.data.txt"
        )
        if os.path.exists(file):
            tmp = process_file(file)
            if tumor == "ACC":
                meth_df = tmp
            else:
                meth_df = pd.concat([meth_df, tmp], axis=0)
    meth_df.drop("Composite Element REF", axis=1, inplace=True)
    meth_df["Hybridization REF"] = meth_df["Hybridization REF"].apply(
        lambda x: x.lower()[:-3]
    )
    meth_df = meth_df.drop_duplicates(subset=["Hybridization REF"])
    tmp_list = np.asarray(list(meth_df))
    meth_df = meth_df[tmp_list[meth_df.isna().sum(axis=0) == 0]]
    meth_df.to_csv(root + "/FINAL/methylation.csv", index=False)


def reduce_dimensionality(root=""):
    os.makedirs(os.path.join(root, "cleaned"), exist_ok=True)
    RPPA = pd.read_csv(os.path.join(root, "FINAL/RPPA.csv"))
    methylation = pd.read_csv(os.path.join(root, "FINAL/methylation.csv"))
    miRNAseq = pd.read_csv(os.path.join(root, "FINAL/miRNAseq_RPKM_log2.csv"))
    mRNAseq = pd.read_csv(os.path.join(root, "FINAL/mRNAseq_RSEM.csv"))

    from sklearn.decomposition import KernelPCA

    for view, df in zip(
            ["RPPA", "miRNAseq", "methylation", "mRNAseq"],
            [RPPA, miRNAseq, methylation, mRNAseq],
    ):
        z_dim = 100
        pca = KernelPCA(kernel="poly", n_components=z_dim, random_state=1234)
        z = pca.fit_transform(np.asarray(df.iloc[:, 1:]))
        df_pca = pd.DataFrame(z, index=df["Hybridization REF"]).reset_index()
        df_pca.to_csv(
            os.path.join(root, "cleaned/{}_kpca.csv".format(view)), index=False
        )


class TGCA(Dataset):
    def __init__(
            self, root, download=False, smoketest=False, preprocess=False, complete=False, mortality=1
    ):
        citation = """TCGA requests that authors who use any data from TCGA (including clinical, molecular, 
        and imaging data) acknowledge the TCGA Research Network in the acknowledgements section of their work. \nAn 
        example of a proper acknowledgement is: \n\nThe results <published or shown> here are in whole or part based 
        upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga. \n\n Citation of original 
        TCGA marker papers producing the data utilized is optional. \n Authors are encouraged to recognize the 
        contribution of the appropriate specimen donors and research groups. \n The TCGA Program requests that 
        journal editors, reviewers, and conference organizers attempt to ascertain if appropriate TCGA 
        acknowledgements are made.) """
        self.resources = (
            "https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/"
        )
        self.mortality = mortality
        self.root = root
        self.modes = [
            "mRNAseq_Preprocess",
            "miRseq_Preprocess",
            "RPPA_AnnotateWithGene",
            "Methylation_Preprocess",
        ]
        self.complete = complete
        if smoketest:
            self.tumor_list = ["ACC", "BLCA", "BRCA"]
        else:
            self.tumor_list = tumor_list
        if download:
            self.download()
        if preprocess:
            self.preprocess()
        # clinical(self.tumor_list, self.raw_folder)
        self.missing_obvs()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )
        print(citation)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_raw_exists(self) -> bool:
        return os.path.exists(os.path.join(self.raw_folder))

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.raw_folder, "FINAL"))

    def missing_obvs(self):
        self.mrna = pd.read_csv(
            os.path.join(self.raw_folder, "cleaned/mRNAseq_kpca.csv")
        ).set_index("Hybridization REF")
        self.mirna = pd.read_csv(
            os.path.join(self.raw_folder, "cleaned/miRNAseq_kpca.csv")
        ).set_index("Hybridization REF")
        self.methylation = pd.read_csv(
            os.path.join(self.raw_folder, "cleaned/methylation_kpca.csv")
        ).set_index("Hybridization REF")
        self.rppa = pd.read_csv(
            os.path.join(self.raw_folder, "cleaned/RPPA_kpca.csv")
        ).set_index("Hybridization REF")
        self.clinical = pd.read_csv(
            os.path.join(self.raw_folder, "FINAL/clinical_label.csv")
        ).set_index("Hybridization REF")
        clinical_idx = self.clinical.loc[
            self.clinical["1yr-mortality"] != -1
            ].index.tolist()
        idx_list_all = np.unique(
            self.mrna.index.tolist()
            + self.mirna.index.tolist()
            + self.rppa.index.tolist()
            + self.methylation.index.tolist()
        )
        self.idx_list = np.intersect1d(idx_list_all, clinical_idx)
        self.idx_list_complete = list(set.intersection(set(self.mrna.index), set(self.mirna.index), set(self.rppa.index),
                                                  set(self.methylation.index), set(clinical_idx)))
        idx_df = pd.DataFrame(index=self.idx_list)
        self.mask=idx_df.copy()
        self.mask[['mrna','mirna','rppa','methylation']]=0
        self.mask.loc[np.intersect1d(self.mask.index,self.mrna.index),'mrna']=1
        self.mask.loc[np.intersect1d(self.mask.index,self.mirna.index),'mirna'] = 1
        self.mask.loc[np.intersect1d(self.mask.index,self.rppa.index),'rppa'] = 1
        self.mask.loc[np.intersect1d(self.mask.index,self.methylation.index),'methylation'] = 1
        self.mrna = idx_df.join(self.mrna)
        self.mirna = idx_df.join(self.mirna)
        self.methylation = idx_df.join(self.methylation)
        self.rppa = idx_df.join(self.rppa)
        self.clinical = idx_df.join(self.clinical[[f"{self.mortality}yr-mortality"]])

    def preprocess(self):
        os.makedirs(os.path.join(self.raw_folder, "FINAL"), exist_ok=True)
        clinical(self.tumor_list, self.raw_folder)
        rppa(self.tumor_list, self.raw_folder)
        mrnaseq(self.tumor_list, self.raw_folder)
        methylation(self.tumor_list, self.raw_folder)
        mirnaseq(self.tumor_list, self.raw_folder)
        reduce_dimensionality(self.raw_folder)

    def download(self) -> None:
        """Download the data if it doesn't exist in processed_folder already."""

        if not self._check_raw_exists():
            os.makedirs(self.raw_folder, exist_ok=True)
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            for tumor in self.tumor_list:
                for mode in self.modes:
                    raw = (
                            self.resources
                            + f"{tumor}/20160128/gdac.broadinstitute.org_{tumor}.{mode}.Level_3.2016012800.0.0.tar.gz"
                    )
                    try:
                        download_and_extract_archive(raw, download_root=self.raw_folder)
                    except:
                        pass
                raw = (
                        self.resources
                        + f"{tumor}/20160128/gdac.broadinstitute.org_{tumor}.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz"
                )
                try:
                    download_and_extract_archive(raw, download_root=self.raw_folder)
                except:
                    pass

    def __len__(self):
        if self.complete:
            return len(self.idx_list_complete)
        else:
            return len(self.idx_list)

    def __getitem__(self, index):
        if self.complete:
            patient_id = self.idx_list_complete[index]
        else:
            patient_id = self.idx_list[index]
        batch = {"index": index, "views": [
            self.mrna.loc[patient_id].values,
            self.mirna.loc[patient_id].values,
            self.rppa.loc[patient_id].values,
            self.methylation.loc[patient_id].values,
        ], "labels": self.clinical.loc[patient_id].values, "mask": self.mask.loc[patient_id]}
        return batch
