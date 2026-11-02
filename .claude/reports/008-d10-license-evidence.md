# Report: SPEC-008 D-10, License and Data-Terms Evidence

**Date:** 2026-11-02
**Prompt:** PROMPT-008D10 (Part A)
**Decision:** D-10, "Confirm UNI terms cover this study and record the SurGen data terms" (`cohort.yaml`, status **pending**, gates `pilot`, `extraction`)
**Outcome:** Evidence recorded. **D-10 is NOT approved by this report.** §5 gives a recommendation for the owner.
**Update (PROMPT-008C2-AUTH, 2026-11-02):** the owner confirmed items 1–4 of §5 and the EBI download route, and D-10 is now approved in `cohort.yaml`. Item 6 (institutional ethics position) was not confirmed and stays OPEN; it blocks publication, not the pilot.

**Constraints honoured:**
- Only public metadata and documentation pages were read: the UNI model card and repository, the SurGen repository, the BioStudies study metadata of S-BIAD1285 (one JSON document, no file listing, no data file) and the EMBL-EBI terms.
- No WSI, checkpoint or embedding was downloaded. UNI was not run.
- Copies of the fetched evidence were kept in the session scratchpad only, not in the repository. Their SHA-256 values are listed so the evidence can be re-checked.

Status legend:
- **CONFIRMED**: quoted from an official source listed in §1
- **LOCAL**: verified in local files
- **INTERPRETATION**: a reading of the terms. It is not legal advice and should be checked by the institution if anything depends on it.
- **OPEN**: cannot be established from any available source. Only the owner can answer it.

---

## 1. Sources (retrieved 2026-11-02, about 22:10 UTC)

| Id | Source | Version / identity | Evidence copy SHA-256 |
|---|---|---|---|
| U1 | Hugging Face model card metadata, `https://huggingface.co/api/models/MahmoodLab/UNI` (public part; the weights are gated) | repo `sha b55a5ec6cade1a39edfe6534189a9b8ca7a022f0`, `lastModified 2025-03-06T22:57:31Z`, `gated: auto` | `33f2bf872c75da0d92e25537294a4f6cbdebb970fd808a388df80a7f53f93bbf` |
| U2 | `github.com/mahmoodlab/UNI`, `README.md`, section "License and Terms of Tuse" [sic], line 466 | commit `42715efc11722a496e0a67f3369505a8f277206c` (2025-03-26), the same commit read in SPEC-006 | `4ac024c83dbcdc39987a81f4983474b0e6c6f15352226677809a6fb492f9cdb8` |
| U3 | `github.com/mahmoodlab/UNI`, `LICENSE` | Full legal text of "Attribution-NonCommercial-NoDerivatives 4.0 International" | `e7e2efbb4575317076f44c8843579a017cb0718dbc4da4f72aff51ec8552ca80` |
| U4 | Local checkpoint `../digital-pathology/data/checkpoints/huggingface/uni.pth` and its `huggingface_hub` download metadata | SHA-256 `56ef09b4…fb40`; downloaded at repo commit `b55a5ec6…` (SPEC-006 report §6) | — |
| S1 | BioStudies study metadata, `https://www.ebi.ac.uk/biostudies/api/v1/studies/S-BIAD1285` | Release date 2024-07-24; dataset DOI `10.6019/S-BIAD1285`; description updated 2026-08-03 (OME-Zarr added) | `4164333eeea2976d2cc8fae0e8bf1f54d4b4baa97bf54203d4da520d814dc59a` |
| S2 | `github.com/CraigMyles/SurGen-Dataset`, `README.md` and `LICENSE` | Local clone at the pinned commit `e56ca8cd…`; current `main` is `55c9bcb7…` (2026-08-29). The licence and citation text is the same in both. | — |
| S3 | Data note: Myles et al., *GigaScience* 14, giaf086 (2025), `doi:10.1093/gigascience/giaf086`; preprint arXiv:2502.04946 | Read through a summarizing fetch, not verbatim | — |
| S4 | EMBL-EBI Terms of Use, `https://www.ebi.ac.uk/about/terms-of-use/` | Read through a summarizing fetch, not verbatim | — |
| S5 | Zenodo record 14047723, "Patch-level UNI feature embeddings … SurGen dataset" (not used by this study; context only) | `license: cc-by-4.0` (API) | — |

---

## 2. UNI (model weights and code)

### 2.1 What the terms say (CONFIRMED)

- **Licence:** `cc-by-nc-nd-4.0` (U1 `cardData.license`; U3 legal text; U2).
- **Gated-access terms**, verbatim from U1 `extra_gated_prompt` (U2 line 466 repeats them almost word for word):

  > This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the UNI model and its derivatives, which include models trained on outputs from the UNI model or datasets created from the UNI model, is prohibited and requires prior approval. Please note that the primary email used to sign up for your Hugging Face account must match your institutional email to receive approval. By downloading the model, you attest that all information (affiliation, research use) is correct and up-to-date. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the UNI model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model. If you are a commercial entity, please contact the corresponding author.

- **Access form fields** (U1 `extra_gated_fields`): full name; current affiliation; type of affiliation (Academia, Industry, Other); institutional email, which "must match your primary email in your Hugging Face account, @gmail/@hotmail/@qq email domains will be denied"; the intended research use; and three agreements (all terms; "non-commercial, academic purposes only"; no distribution, and individual registration for other users in the organization).
- **Attribution:** "with proper attribution". The citation the authors ask for (U2): Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F.K., et al. *Towards a general-purpose foundation model for computational pathology.* Nat Med (2024). `https://doi.org/10.1038/s41591-024-02857-3`.

### 2.2 Obligations and restrictions for this study

| # | Obligation or restriction | Source | How this repository meets it | Status |
|---|---|---|---|---|
| U-a | Non-commercial, academic research only | U1, U2, U3 | The intended use is academic research (§4) | CONFIRMED term; use is an ASSUMPTION until the owner confirms it |
| U-b | Access only through an approved, individual Hugging Face registration with an institutional email, and an accurate statement of the research use | U1 | The checkpoint was downloaded at repo commit `b55a5ec6…` (U4). **Which account downloaded it, whether that account was approved with an institutional email, and what research use it declared are not recorded anywhere locally.** | **OPEN** |
| U-c | Each other person in the organization registers individually | U1, U2 | The checkpoint is used in place from a sibling repository on one personal Mac. Anyone else who runs the pilot needs their own approval. | OPEN for any second operator |
| U-d | Do not distribute, publish or reproduce a copy of the model | U1, U2 | The checkpoint is git-ignored in the legacy repository, used read-only by path, never copied into this repository (SPEC-006 §6). The C2 binding records only its SHA-256. | LOCAL |
| U-e | Derivatives ("models trained on outputs from the UNI model or datasets created from the UNI model") may not be used commercially | U1, U2 | UNI embedding stores and any MIL model trained on them are such derivatives. Phase C2 creates embeddings only, and only for research. | CONFIRMED term |
| U-f | CC BY-NC-ND 4.0 does not permit sharing Adapted Material | U3 §2(a)(1)(B), §3 | Embeddings are written outside the repository (`~/pathology-ai-data`, mode 0700). Nothing in the plan publishes them. Whether an embedding is "Adapted Material" in the copyright sense is unsettled (INTERPRETATION). The conservative course is to treat embeddings and UNI-based MIL weights as **not shareable**. | INTERPRETATION |
| U-g | Do not try to re-identify UNI's training data | U1, U2 | Not applicable to this work; nothing does so | CONFIRMED term |
| U-h | Attribution | U1, U2, U3 | Every UNI manifest already carries the licence statement in `limitations` (SPEC-006). Publications must cite Chen et al. 2024. | LOCAL (manifests); future publications must comply |

### 2.3 Publication of results (INTERPRETATION)

- Publishing aggregate results (AUROC, confusion matrices, methods) obtained with UNI features is the ordinary academic use the terms describe. It is how the authors' own benchmarks are reported.
- Publishing the embeddings, the UNI-derived MIL weights, or a tool that bundles them is a distribution of derivatives. The terms do not clearly permit that, so it should be excluded unless the authors approve it.
- S5 shows that the SurGen authors published UNI embeddings under CC BY 4.0. That is their arrangement, not evidence that this study may do the same.

---

## 3. SurGen data (S-BIAD1285)

### 3.1 What the terms say

| Item | Evidence | Status |
|---|---|---|
| Dataset licence | S1 attribute `License = "CC BY 4.0"`, qualifier `URL = https://creativecommons.org/licenses/by/4.0/legalcode` | CONFIRMED |
| Release and DOI | `ReleaseDate 2024-07-24`; `DOI 10.6019/S-BIAD1285` | CONFIRMED |
| Creators | Craig Myles (first author), In Hwa Um, Craig Marshall (data acquisition), David Harris-Birtill, David J Harrison (principal investigators); University of St Andrews; NHS Lothian Biorepository | CONFIRMED (S1) |
| Acknowledgement requested | "We acknowledge the support of NHS Research Scotland (NRS) Lothian Biorepository." | CONFIRMED (S1) |
| Funding statement | iCAIRD, funded by Innovate UK on behalf of UKRI, project 104690 | CONFIRMED (S1) |
| Access conditions, data-use agreement, ethics or consent attributes | **None** in the study metadata | CONFIRMED (absent from S1) |
| File checksums | **None** in the study metadata. EBI publishes no digest that the 5 downloads could be checked against, so their SHA-256 is "trust on first download". | CONFIRMED (absent from S1) |
| Label CSVs (`reproducibility/dataset_csv`, the D-1 label authority) | "dedicated to the public domain under **CC0**" | CONFIRMED (S2 README) |
| Repository code | GPL-3.0 (S2 `LICENSE`). This study does not copy or link the upstream code: it only reads the pinned CSVs. | CONFIRMED / LOCAL |
| Citation requested | Myles C, Um IH, Marshall C, Harris-Birtill D, Harrison DJ. *SurGen: 1020 H&E-stained whole-slide images with survival and genetic markers.* GigaScience 14, giaf086 (2025). The README also lists Myles et al., MIUA 2024 ("Leveraging foundation models for enhanced detection of colorectal cancer biomarkers in small datasets"). | CONFIRMED (S2) |
| Ethics of the original collection | The data note reports approval by the University of St Andrews School of Computer Science Ethics Committee (CS16553) and the Lothian NRS BioResource RTB (REC 20/ES/0061 and 13/ES/0126), and says the data were de-identified | S3, from a summarizing fetch: **NOT VERIFIED verbatim** |
| Repository terms | EMBL-EBI "places no additional restrictions on the use or redistribution of the data … other than those provided by the original data owners"; it expects attribution and provides data "AS IS"; service-abuse rules apply to access | S4, from a summarizing fetch |
| Dataset change since the study design | On 2026-08-03 the study added OME-Zarr copies. Whether the original CZI files changed is not stated. The acquisition records ETag and Last-Modified, and the 3 local slides are checked against the C1 inventory. | CONFIRMED (S1 description); CZI immutability NOT VERIFIED |

### 3.2 Obligations for this study

| # | Obligation | When | Status |
|---|---|---|---|
| S-a | CC BY 4.0 attribution (creators, licence notice and link, indication of changes) | When SurGen material, or material adapted from it, is **shared** (CC BY 4.0 §3(a)). Private research use carries no condition. | CONFIRMED term |
| S-b | Cite the dataset DOI `10.6019/S-BIAD1285` and the data note in any publication. Include the NRS Lothian Biorepository acknowledgement. | Publications | CONFIRMED (S1, S2) |
| S-c | No additional restriction from EMBL-EBI. Downloads use the documented public HTTPS route at a moderate rate (5 files, sequential, one connection). | Phase C2 | CONFIRMED (S4) |
| S-d | Do not attempt to re-identify patients | Always | INTERPRETATION: no SurGen term says so, but it follows from the de-identified release and ordinary research ethics |
| S-e | Institutional ethics position for secondary analysis of public, de-identified human data | Before publication, possibly before use | **OPEN**: a question for the owner's institution, not for the data licence |

---

## 4. Is the intended use permitted?

The intended use, as recorded in SPEC-008, ADR-008 and D-9:
- local, academic, non-commercial research by the repository owner;
- 8 SurGen WSIs (Phase C2), later the eligible cohort (Phase D);
- UNI `uni-official@1` embeddings computed on the owner's Mac;
- MIL training and evaluation for MSI/dMMR;
- publication of aggregate results;
- no redistribution of weights, WSIs or embeddings.

That the use is academic and non-commercial is an ASSUMPTION drawn from the repository (`pyproject.toml`: "research use only"). It is not a recorded owner statement.

| Question | Answer | Basis |
|---|---|---|
| May SurGen WSIs be downloaded, stored locally and analysed? | **Yes** | CC BY 4.0 (S1) and no extra EBI restriction (S4). CONFIRMED. |
| May the SurGen label CSVs be used and redistributed? | **Yes**, without condition | CC0 (S2). CONFIRMED. |
| May UNI be used to embed SurGen for this academic study? | **Yes, if** U-b holds: approved individual access under an institutional email, with an accurate research-use statement | U1/U2 (CONFIRMED terms); U-b is **OPEN** |
| May the results be published? | Yes, with UNI and SurGen attribution | INTERPRETATION (§2.3); S-b CONFIRMED |
| May UNI embeddings or UNI-based MIL weights be shared or published? | **Treat as no**, unless the UNI authors approve it | U-e/U-f, INTERPRETATION |
| May any of this be used commercially? | **No** for anything derived from UNI | CONFIRMED |
| Does anything in the SurGen terms block Phase C2? | **No** | CONFIRMED |

**Confirmed vs assumed:**
- **Confirmed:** both licences and their texts; the gated-access conditions; the absence of a SurGen data-use agreement and of published checksums; the citations; the checkpoint's identity (`56ef09b4…`, repo commit `b55a5ec6…`).
- **Assumed:**
  - the study is non-commercial academic research;
  - the checkpoint was obtained through an approved individual registration whose declared use covers this study;
  - the owner is the only operator;
  - embeddings count as UNI derivatives and should not be shared;
  - no institutional ethics review is needed before Phase C2.

---

## 5. Recommendation to the owner (not a decision)

**Recommendation:** D-10 can be approved for the Phase C2 pilot **if the owner can truthfully confirm items 1–3 below**. If item 1 cannot be confirmed, D-10 should stay pending. Valid access would then have to be obtained first: a new approved registration and, if needed, a re-download checked against the pin `56ef09b4…`.

The owner confirms, in the decision record:
1. **UNI access (U-b).** Name the Hugging Face account that downloaded the checkpoint, say that its access request was approved, and say that its primary email was the institutional one that was declared. The terms deny access to `@gmail`, `@hotmail` and `@qq` primary emails. Also say whether the declared research use covers SurGen MSI/dMMR MIL research.
2. **Purpose.** Non-commercial academic research; no commercial use of UNI derivatives.
3. **Operator.** Only the registered individual runs UNI; any other operator registers individually first.
4. **Distribution.** No UNI weights, embeddings or UNI-derived model weights leave the owner's control. Results are published as aggregates only.
5. **Attribution.** Cite UNI (Chen et al. 2024), SurGen (DOI `10.6019/S-BIAD1285` and GigaScience giaf086) and acknowledge NRS Lothian Biorepository. Keep the licence notices in the manifests.
6. **Ethics.** State the institution's position on secondary use of public, de-identified data, or record that it is pending and blocks publication only.

A possible ledger text, if the owner agrees (wording and id are the owner's):

```yaml
  - id: D-10
    summary: >-
      Licences reviewed (.claude/reports/008-d10-license-evidence.md). UNI (CC-BY-NC-ND 4.0,
      gated): used for non-commercial academic research only, through the owner's approved
      individual Hugging Face access (<account / institution / approval date>); no
      redistribution of the weights, of UNI embeddings or of UNI-derived models; attribution
      to Chen et al. 2024. SurGen S-BIAD1285 (CC BY 4.0; label CSVs CC0): attribution to the
      dataset DOI and data note, NRS Lothian acknowledgement. No commercial use.
    status: approved
    gates: [pilot, extraction]
    approver: ...
    approved_on: "YYYY-MM-DD"
    reference: ...
```

Consequences of recording it:
- G5 of the C2 preflight passes.
- The data-free test `test_configuration_differs_from_v1_only_by_the_recorded_approvals` pins `pending == {"D-10"}`, so it must be updated in the same change.
- Manifest v1 is not affected: its ledger keeps D-10 as pending, as it was when v1 was built.
