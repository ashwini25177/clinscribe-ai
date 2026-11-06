"""
ClinScribe AI — HL7 FHIR R4 Exporter
======================================
Converts clinical speech-to-insight pipeline results into interoperable
HL7 FHIR Release 4 JSON resources:
  - Bundle (document / collection)
  - Encounter (clinical consultation record)
  - DocumentReference (SOAP note / clinical summary)
  - Condition (extracted diagnoses, symptoms, medical history)
  - MedicationRequest (extracted medications, dosages, frequency, route)
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any


class FHIRR4Exporter:
    """
    Exports ClinScribe pipeline outputs into HL7 FHIR R4 compliant JSON resources.
    """

    def __init__(self, system_uri: str = "https://clinscribe.ai/fhir"):
        self.system_uri = system_uri

    def to_bundle(
        self,
        pipeline_output: Dict[str, Any],
        patient_id: str = "PATIENT-DEMO-001",
        practitioner_id: str = "PRACTITIONER-001",
    ) -> Dict[str, Any]:
        """
        Convert complete pipeline output into an HL7 FHIR R4 Bundle.

        Args:
            pipeline_output: Result dict from ClinScribePipeline containing
                             transcript, entities, and summary/soap.
            patient_id: Identifier of the patient.
            practitioner_id: Identifier of the clinician.

        Returns:
            FHIR R4 Bundle dictionary.
        """
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        bundle_id = f"clinscribe-bundle-{uuid.uuid4().hex[:8]}"
        encounter_id = f"enc-{uuid.uuid4().hex[:8]}"

        entries = []

        # 1. Encounter resource
        encounter = self._build_encounter(encounter_id, patient_id, practitioner_id, now_iso)
        entries.append({
            "fullUrl": f"urn:uuid:{encounter_id}",
            "resource": encounter
        })

        # 2. DocumentReference resource for SOAP note
        summary = pipeline_output.get("summary", {})
        soap = summary.get("soap") if isinstance(summary, dict) else None
        summary_text = summary.get("summary", "") if isinstance(summary, dict) else str(summary)
        transcript = pipeline_output.get("transcript", "")

        doc_ref = self._build_document_reference(
            patient_id=patient_id,
            encounter_id=encounter_id,
            transcript=transcript,
            summary_text=summary_text,
            soap=soap,
            timestamp=now_iso,
        )
        entries.append({
            "fullUrl": f"urn:uuid:{doc_ref['id']}",
            "resource": doc_ref
        })

        # 3. Condition resources
        raw_entities = []
        if isinstance(pipeline_output.get("entities"), dict):
            raw_entities = pipeline_output["entities"].get("raw", [])
        elif isinstance(pipeline_output.get("entities"), list):
            raw_entities = pipeline_output["entities"]

        conditions = [
            e for e in raw_entities
            if any(k in e.get("label", "").lower() for k in ["condition", "disease", "history", "symptom"])
        ]
        for cond in conditions:
            cond_res = self._build_condition(patient_id, encounter_id, cond, now_iso)
            entries.append({
                "fullUrl": f"urn:uuid:{cond_res['id']}",
                "resource": cond_res
            })

        # 4. MedicationRequest resources
        medications = [
            e for e in raw_entities
            if any(k in e.get("label", "").lower() for k in ["medication", "drug"])
        ]
        dosages = [
            e for e in raw_entities
            if "dosage" in e.get("label", "").lower()
        ]
        routes = [
            e for e in raw_entities
            if any(k in e.get("label", "").lower() for k in ["route", "administration"])
        ]

        # Associate dosages and routes with medications
        for i, med in enumerate(medications):
            assoc_dosage = dosages[i]["text"] if i < len(dosages) else None
            assoc_route = routes[i]["text"] if i < len(routes) else None
            med_res = self._build_medication_request(
                patient_id=patient_id,
                encounter_id=encounter_id,
                med_entity=med,
                dosage_text=assoc_dosage,
                route_text=assoc_route,
                timestamp=now_iso,
            )
            entries.append({
                "fullUrl": f"urn:uuid:{med_res['id']}",
                "resource": med_res
            })

        return {
            "resourceType": "Bundle",
            "id": bundle_id,
            "type": "collection",
            "timestamp": now_iso,
            "total": len(entries),
            "entry": entries
        }

    def _build_encounter(
        self,
        encounter_id: str,
        patient_id: str,
        practitioner_id: str,
        timestamp: str
    ) -> Dict[str, Any]:
        return {
            "resourceType": "Encounter",
            "id": encounter_id,
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "AMB",
                "display": "ambulatory"
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "participant": [
                {
                    "individual": {"reference": f"Practitioner/{practitioner_id}"}
                }
            ],
            "period": {"start": timestamp, "end": timestamp}
        }

    def _build_document_reference(
        self,
        patient_id: str,
        encounter_id: str,
        transcript: str,
        summary_text: str,
        soap: Optional[Dict[str, str]],
        timestamp: str
    ) -> Dict[str, Any]:
        formatted_content = ""
        if soap and any(soap.values()):
            formatted_content += "CLINICAL SOAP NOTE:\n"
            for k in ["S", "O", "A", "P"]:
                if soap.get(k):
                    title = {"S": "Subjective", "O": "Objective", "A": "Assessment", "P": "Plan"}.get(k, k)
                    formatted_content += f"\n[{title}]:\n{soap[k]}\n"
        elif summary_text:
            formatted_content += f"CLINICAL SUMMARY:\n{summary_text}\n"

        if transcript:
            formatted_content += f"\nRAW CONSULTATION TRANSCRIPT:\n{transcript}"

        return {
            "resourceType": "DocumentReference",
            "id": f"doc-{uuid.uuid4().hex[:8]}",
            "status": "current",
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11488-4",
                        "display": "Consultation note"
                    }
                ],
                "text": "Clinical Consultation & SOAP Note"
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "context": {
                "encounter": [{"reference": f"Encounter/{encounter_id}"}]
            },
            "date": timestamp,
            "content": [
                {
                    "attachment": {
                        "contentType": "text/plain",
                        "data": None,
                        "title": "Clinical Summary Note",
                        "creation": timestamp
                    },
                    "format": {
                        "system": "http://ihe.net/fhir/ValueSet/IHE.FormatCode.codesystem",
                        "code": "urn:ihe:pcc:ic:2008",
                        "display": "Immunization Details"
                    }
                }
            ],
            "description": formatted_content.strip()
        }

    def _build_condition(
        self,
        patient_id: str,
        encounter_id: str,
        condition_entity: Dict[str, Any],
        timestamp: str
    ) -> Dict[str, Any]:
        cond_text = condition_entity.get("text", "Unknown Condition")
        score = condition_entity.get("score", 1.0)
        return {
            "resourceType": "Condition",
            "id": f"cond-{uuid.uuid4().hex[:8]}",
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                        "display": "Active"
                    }
                ]
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "confirmed" if score > 0.8 else "provisional",
                        "display": "Confirmed" if score > 0.8 else "Provisional"
                    }
                ]
            },
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                            "code": "encounter-diagnosis",
                            "display": "Encounter Diagnosis"
                        }
                    ]
                }
            ],
            "code": {
                "text": cond_text
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "encounter": {"reference": f"Encounter/{encounter_id}"},
            "recordedDate": timestamp
        }

    def _build_medication_request(
        self,
        patient_id: str,
        encounter_id: str,
        med_entity: Dict[str, Any],
        dosage_text: Optional[str],
        route_text: Optional[str],
        timestamp: str
    ) -> Dict[str, Any]:
        drug_name = med_entity.get("text", "Unknown Drug")
        resource = {
            "resourceType": "MedicationRequest",
            "id": f"medrx-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {
                "text": drug_name
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "encounter": {"reference": f"Encounter/{encounter_id}"},
            "authoredOn": timestamp,
            "dosageInstruction": []
        }

        instruction = {}
        if dosage_text:
            instruction["text"] = f"Take {dosage_text}"
        if route_text:
            instruction["route"] = {"text": route_text}

        if instruction:
            resource["dosageInstruction"].append(instruction)

        return resource
