#!/usr/bin/env python
import json
from jarvis.db.figshare import data


def collect_data():
    # def collect_data(dft_3d, vacancydb, surface_data):
    dft_3d = data("dft_3d")
    vacancydb = data("vacancydb")
    surface_data = data("surfacedb")
    defect_ids = list(set([entry["jid"] for entry in vacancydb]))
    surf_ids = list(
        set(
            [
                entry["name"].split("Surface-")[1].split("_miller_")[0]
                for entry in surface_data
            ]
        )
    )

    aggregated_data = []
    for entry in dft_3d:
        tmp = entry
        tmp["vacancy"] = {}
        tmp["surface"] = {}

        # Check if the entry is in the defect dataset
        if entry["jid"] in defect_ids:
            for vac_entry in vacancydb:
                if entry["jid"] == vac_entry["jid"]:
                    tmp["vacancy"].setdefault(
                        vac_entry["id"].split("_")[0]
                        + "_"
                        + vac_entry["id"].split("_")[1],
                        vac_entry["ef"],
                    )

        # Check if the entry is in the surface dataset
        if entry["jid"] in surf_ids:
            for surf_entry in surface_data:
                jid = (
                    surf_entry["name"]
                    .split("Surface-")[1]
                    .split("_miller_")[0]
                )
                if entry["jid"] == jid:
                    tmp["surface"].setdefault(
                        "_".join(
                            surf_entry["name"]
                            .split("_thickness")[0]
                            .split("_")[0:5]
                        ),
                        surf_entry["surf_en"],
                    )

        aggregated_data.append(tmp)

    return aggregated_data


def get_vacancy_energy_entry(jid, aggregated_data):
    """
    Retrieve the vacancy formation energy entry (vac_en_entry) for a given jid.

    Parameters:
    jid (str): The JID of the material.
    aggregated_data (list): The aggregated data containing vacancy and surface information.

    Returns:
    dict: A dictionary containing the vacancy formation energy entry and corresponding symbol.
    """
    for entry in aggregated_data:
        if entry["jid"] == jid:
            vacancy_data = entry.get("vacancy", {})
            if vacancy_data:
                return [
                    {"symbol": key, "vac_en_entry": value}
                    for key, value in vacancy_data.items()
                ]
            else:
                return f"No vacancy data found for JID {jid}"
    return f"JID {jid} not found in the data."


def get_surface_energy_entry(jid, aggregated_data):
    """
    Retrieve the surface energy entry (surf_en_entry) for a given jid.

    Parameters:
    jid (str): The JID of the material.
    aggregated_data (list): The aggregated data containing vacancy and surface information.

    Returns:
    list: A list of dictionaries containing the surface energy entry and corresponding name.
    """
    for entry in aggregated_data:
        if entry["jid"] == jid:
            surface_data = entry.get("surface", {})
            if surface_data:
                # Prepend 'Surface-JVASP-<jid>_' to the key for correct matching
                return [
                    {"name": f"{key}", "surf_en_entry": value}
                    for key, value in surface_data.items()
                ]
            else:
                return f"No surface data found for JID {jid}"
    return f"JID {jid} not found in the data."


def log_job_info(message, log_file):
    """Log job information to a file and print it."""
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)


def save_dict_to_json(data_dict, filename):
    with open(filename, "w") as f:
        json.dump(data_dict, f, indent=4)


def load_dict_from_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_jid_list():
    jids = [
        "JVASP-8184",
        "JVASP-10591",
        "JVASP-8118",
        "JVASP-8003",
        "JVASP-1222",
        "JVASP-106363",
        "JVASP-1109",
        "JVASP-96",
        "JVASP-20092",
        "JVASP-30",
        "JVASP-1372",
        "JVASP-23",
        "JVASP-105410",
        "JVASP-36873",
        "JVASP-113",
        "JVASP-7836",
        "JVASP-861",
        "JVASP-9117",
        "JVASP-108770",
        "JVASP-9147",
        "JVASP-1180",
        "JVASP-10703",
        "JVASP-79522",
        "JVASP-21211",
        "JVASP-1195",
        "JVASP-8082",
        "JVASP-1186",
        "JVASP-802",
        "JVASP-8559",
        "JVASP-14968",
        "JVASP-43367",
        "JVASP-22694",
        "JVASP-3510",
        "JVASP-36018",
        "JVASP-90668",
        "JVASP-110231",
        "JVASP-149916",
        "JVASP-1103",
        "JVASP-1177",
        "JVASP-1115",
        "JVASP-1112",
        "JVASP-25",
        "JVASP-10037",
        "JVASP-103127",
        "JVASP-813",
        "JVASP-1067",
        "JVASP-825",
        "JVASP-14616",
        "JVASP-111005",
        "JVASP-1002",
        "JVASP-99732",
        "JVASP-54",
        "JVASP-133719",
        "JVASP-1183",
        "JVASP-62940",
        "JVASP-14970",
        "JVASP-34674",
        "JVASP-107",
        "JVASP-58349",
        "JVASP-110",
        "JVASP-1915",
        "JVASP-816",
        "JVASP-867",
        "JVASP-34249",
        "JVASP-1216",
        "JVASP-32",
        "JVASP-1201",
        "JVASP-2376",
        "JVASP-18983",
        "JVASP-943",
        "JVASP-104764",
        "JVASP-39",
        "JVASP-10036",
        "JVASP-1312",
        "JVASP-8554",
        "JVASP-1174",
        "JVASP-8158",
        "JVASP-131",
        "JVASP-36408",
        "JVASP-85478",
        "JVASP-972",
        "JVASP-106686",
        "JVASP-1008",
        "JVASP-4282",
        "JVASP-890",
        "JVASP-1192",
        "JVASP-91",
        "JVASP-104",
        "JVASP-963",
        "JVASP-1189",
        "JVASP-149871",
        "JVASP-5224",
        "JVASP-41",
        "JVASP-1240",
        "JVASP-1408",
        "JVASP-1023",
        "JVASP-1029",
        "JVASP-149906",
        "JVASP-1327",
        "JVASP-29539",
        "JVASP-19780",
        "JVASP-85416",
        "JVASP-9166",
        "JVASP-1198",
    ]
    return jids
