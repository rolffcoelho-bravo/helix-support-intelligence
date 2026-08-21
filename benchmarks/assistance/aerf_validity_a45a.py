"""Candidate-independent AERF validity construction for Phase 4 A4.5a.

This module performs no learned inference. It builds a fresh fictional auxiliary
support corpus and gold AERF component/claim cases before any model binding.
"""
from __future__ import annotations
import hashlib, json
from typing import Any

SEED = 20260821
CORPUS_ID = "helix-aerf-validity-corpus-v1"
PROTOCOL_ID = "phase4-assistance-a4.5a-aerf-validity-v1"
QUEUES = ("access_review","payments_review","transfers_review","cash_operations","identity_review","account_services")
REQUIREMENTS = ("identity confirmation","transaction reference","device confirmation","recipient details","statement excerpt","account ownership proof")
WINDOWS = (1,2,3,4,5,7)

def _order(unit_ids:list[str])->list[str]:
    return sorted(unit_ids,key=lambda value: hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest())

def build_units()->list[dict[str,Any]]:
    units=[]
    for index in range(60):
        number=index+1; unit_id=f"AERF-U{number:03d}"; queue=QUEUES[index%len(QUEUES)]
        requirement=REQUIREMENTS[(index*5+1)%len(REQUIREMENTS)]; window=WINDOWS[(index*7+2)%len(WINDOWS)]
        alt_queue=QUEUES[(index+1)%len(QUEUES)]
        subject=f"Orchid case {number:03d}"
        support_queue=f"{subject} requests are handled by the {queue} queue."
        support_requirement=f"{subject} review requires {requirement}."
        support_window=f"The standard review window for {subject.lower()} is {window} business days."
        units.append({"unit_id":unit_id,"subject":subject,"queue":queue,"alternate_queue":alt_queue,"requirement":requirement,"window_days":window,
            "documents":{
                "queue":{"document_id":f"{unit_id}-Q","text":support_queue},
                "requirement":{"document_id":f"{unit_id}-R","text":support_requirement},
                "window":{"document_id":f"{unit_id}-W","text":support_window},
                "refutation":{"document_id":f"{unit_id}-X","text":f"{subject} requests are not handled by the {queue} queue. They are handled by the {alt_queue} queue."},
                "contaminated":{"document_id":f"{unit_id}-C","text":f"{support_queue} {subject} requests are not automatically approved without review."}}})
    return units

def split_units(units:list[dict[str,Any]])->dict[str,list[str]]:
    ordered=_order([str(u["unit_id"]) for u in units]); calibration=ordered[:40]; validation=ordered[40:]
    if len(calibration)!=40 or len(validation)!=20 or set(calibration)&set(validation): raise RuntimeError("invalid A4.5a split")
    return {"calibration":calibration,"validation":validation}

def _pair(pair_id,split,unit_id,subtype,claim,evidence_document_id,evidence_text,relevance,sufficiency,polarity,final_relation):
    minimal = None if relevance == "IRRELEVANT" else evidence_text.split(". ", 1)[0].rstrip(".") + "."
    return {"pair_id":pair_id,"split":split,"unit_id":unit_id,"subtype":subtype,"claim":claim,"evidence_document_id":evidence_document_id,"evidence_text":evidence_text,"gold":{"relevance":relevance,"sufficiency":sufficiency,"polarity":polarity,"final_relation":final_relation,"minimal_evidence_text":minimal}}

def _claim(case_id,split,unit_id,category,atom_relations,expected_verdict,deterministic_gate="NONE"):
    return {"case_id":case_id,"split":split,"unit_id":unit_id,"category":category,"atom_relations":atom_relations,"deterministic_gate":deterministic_gate,"expected_verdict":expected_verdict}

def build_suite()->dict[str,Any]:
    units=build_units(); by_id={str(u["unit_id"]):u for u in units}; split=split_units(units); pair_rows=[]; claim_rows=[]
    for split_name in ("calibration","validation"):
        ids=split[split_name]
        for offset,unit_id in enumerate(ids):
            u=by_id[unit_id]; d=u["documents"]; subject=str(u["subject"]); queue=str(u["queue"]); req=str(u["requirement"]); window=int(u["window_days"])
            other=by_id[ids[(offset+1)%len(ids)]]; od=other["documents"]["queue"]; p=f"{split_name[:3].upper()}-{unit_id}"
            pair_rows += [
                _pair(f"{p}-P01",split_name,unit_id,"literal_support",f"{subject} requests are handled by the {queue} queue.",d["queue"]["document_id"],d["queue"]["text"],"RELEVANT","SUFFICIENT","SUPPORTS","ENTAILED"),
                _pair(f"{p}-P02",split_name,unit_id,"paraphrase_support",f"The {queue} queue handles {subject.lower()} requests.",d["queue"]["document_id"],d["queue"]["text"],"RELEVANT","SUFFICIENT","SUPPORTS","ENTAILED"),
                _pair(f"{p}-P03",split_name,unit_id,"explicit_refutation",f"{subject} requests are handled by the {queue} queue.",d["refutation"]["document_id"],d["refutation"]["text"],"RELEVANT","SUFFICIENT","REFUTES","CONTRADICTED"),
                _pair(f"{p}-P04",split_name,unit_id,"attribute_refutation",f"{subject} requests are not handled by the {queue} queue.",d["queue"]["document_id"],d["queue"]["text"],"RELEVANT","SUFFICIENT","REFUTES","CONTRADICTED"),
                _pair(f"{p}-P05",split_name,unit_id,"cross_document_irrelevance",f"{subject} requests are handled by the {queue} queue.",od["document_id"],od["text"],"IRRELEVANT","NOT_APPLICABLE","NOT_APPLICABLE","UNKNOWN"),
                _pair(f"{p}-P06",split_name,unit_id,"same_domain_irrelevance",f"{subject} review requires {req}.",d["window"]["document_id"],d["window"]["text"],"IRRELEVANT","NOT_APPLICABLE","NOT_APPLICABLE","UNKNOWN"),
                _pair(f"{p}-P07",split_name,unit_id,"relevant_but_insufficient",f"{subject} requests are handled by the {queue} queue and review requires {req}.",d["queue"]["document_id"],d["queue"]["text"],"RELEVANT","INSUFFICIENT","UNRESOLVED","UNKNOWN"),
                _pair(f"{p}-P08",split_name,unit_id,"temporal_insufficiency",f"{subject} review always finishes within {window} business days.",d["window"]["document_id"],d["window"]["text"],"RELEVANT","INSUFFICIENT","UNRESOLVED","UNKNOWN"),
                _pair(f"{p}-P09",split_name,unit_id,"context_contamination_support",f"{subject} requests are handled by the {queue} queue.",d["contaminated"]["document_id"],d["contaminated"]["text"],"RELEVANT","SUFFICIENT","SUPPORTS","ENTAILED")]
            claim_rows += [
                _claim(f"{p}-C01",split_name,unit_id,"single_supported",["ENTAILED"],"SUPPORTED"),
                _claim(f"{p}-C02",split_name,unit_id,"single_refuted",["CONTRADICTED"],"UNSUPPORTED"),
                _claim(f"{p}-C03",split_name,unit_id,"single_unknown",["UNKNOWN"],"UNSUPPORTED"),
                _claim(f"{p}-C04",split_name,unit_id,"multi_document_supported",["ENTAILED","ENTAILED"],"SUPPORTED"),
                _claim(f"{p}-C05",split_name,unit_id,"partial_multi_document_unsupported",["ENTAILED","UNKNOWN"],"UNSUPPORTED"),
                _claim(f"{p}-C06",split_name,unit_id,"support_refute_conflict",["ENTAILED","CONTRADICTED"],"CONFLICTING_EVIDENCE"),
                _claim(f"{p}-C07",split_name,unit_id,"citation_invalid",["ENTAILED"],"CITATION_INVALID","CITATION_INVALID"),
                _claim(f"{p}-C08",split_name,unit_id,"stale_evidence",["ENTAILED"],"STALE_EVIDENCE","STALE_EVIDENCE"),
                _claim(f"{p}-C09",split_name,unit_id,"registered_conflict",["ENTAILED"],"CONFLICTING_EVIDENCE","REGISTERED_CONFLICT")]
    pair_rows.sort(key=lambda r:str(r["pair_id"])); claim_rows.sort(key=lambda r:str(r["case_id"]))
    return {"protocol_id":PROTOCOL_ID,"corpus_id":CORPUS_ID,"split":split,"units":units,"pair_rows":pair_rows,"claim_rows":claim_rows}

def _jsonl(rows):
    return b"".join((json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n").encode() for r in rows)

def manifest()->dict[str,Any]:
    s=build_suite(); pairs=s["pair_rows"]; claims=s["claim_rows"]; units=s["units"]
    rel={sp:{r:sum(x["split"]==sp and x["gold"]["final_relation"]==r for x in pairs) for r in ("ENTAILED","CONTRADICTED","UNKNOWN")} for sp in ("calibration","validation")}
    return {"protocol_id":PROTOCOL_ID,"corpus_id":CORPUS_ID,"seed":SEED,
      "counts":{"units":len(units),"calibration_units":40,"validation_units":20,"pair_rows":len(pairs),"pair_rows_by_split":{sp:sum(x["split"]==sp for x in pairs) for sp in ("calibration","validation")},"claim_rows":len(claims),"claim_rows_by_split":{sp:sum(x["split"]==sp for x in claims) for sp in ("calibration","validation")},"relation_counts_by_split":rel},
      "sha256":{"units":hashlib.sha256(_jsonl(units)).hexdigest(),"pair_rows":hashlib.sha256(_jsonl(pairs)).hexdigest(),"claim_rows":hashlib.sha256(_jsonl(claims)).hexdigest(),"calibration_pairs":hashlib.sha256(_jsonl([x for x in pairs if x["split"]=="calibration"])).hexdigest(),"validation_pairs":hashlib.sha256(_jsonl([x for x in pairs if x["split"]=="validation"])).hexdigest(),"calibration_claims":hashlib.sha256(_jsonl([x for x in claims if x["split"]=="calibration"])).hexdigest(),"validation_claims":hashlib.sha256(_jsonl([x for x in claims if x["split"]=="validation"])).hexdigest()},
      "candidate_model_calls":0,"confirmatory_query_records_inspected":0,"a44d_rows_reused":0}

if __name__=="__main__": print(json.dumps(manifest(),indent=2,sort_keys=True))