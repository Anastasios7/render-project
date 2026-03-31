from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

from engineapp1 import (
    geo_address_to_lonlat_with_report,
    postcode_from_lonlat_int,
    run_all,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PropertyRequest(BaseModel):
    geo_street: str
    geo_number: str
    geo_city: str
    geo_postcode: str
    geo_country: str
    asset_value: float
    year_construction: int
    floors_house: int
    stability_number: str
    material_house: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="el">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Property Analysis Tool</title>
      <style>
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: Arial, sans-serif;
          background: #f6f7fb;
          color: #1f2937;
        }
        .wrap {
          max-width: 1100px;
          margin: 32px auto;
          padding: 20px;
        }
        .card {
          background: #fff;
          border-radius: 16px;
          box-shadow: 0 10px 30px rgba(0,0,0,0.08);
          padding: 24px;
        }
        h1 {
          margin: 0 0 8px;
          font-size: 28px;
        }
        .subtitle {
          margin: 0 0 24px;
          color: #6b7280;
        }
        .grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
        }
        .field {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        label {
          font-size: 14px;
          font-weight: 700;
        }
        input, select, button {
          width: 100%;
          padding: 12px 14px;
          border: 1px solid #d1d5db;
          border-radius: 10px;
          font-size: 15px;
          background: #fff;
        }
        button {
          cursor: pointer;
          border: 0;
          background: #111827;
          color: white;
          font-weight: 700;
          margin-top: 24px;
        }
        .result {
          margin-top: 24px;
          padding: 18px;
          background: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          min-height: 140px;
          white-space: pre-wrap;
          line-height: 1.6;
        }
        .metrics {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 16px;
          margin-top: 20px;
        }
        .metric {
          background: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          padding: 16px;
        }
        .metric h3 {
          margin: 0 0 8px;
          font-size: 14px;
          color: #6b7280;
        }
        .metric p {
          margin: 0;
          font-size: 22px;
          font-weight: 700;
        }
        .total {
          margin-top: 24px;
          padding: 20px;
          border-radius: 12px;
          background: #ecfdf5;
          border: 1px solid #a7f3d0;
          font-size: 22px;
          font-weight: 700;
        }
        @media (max-width: 768px) {
          .grid, .metrics {
            grid-template-columns: 1fr;
          }
        }
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="card">
          <h1>🏠 Εφαρμογή Εκτίμησης & Geocoding</h1>
          <p class="subtitle">Συμπλήρωσε τα στοιχεία του ακινήτου και πάτα Υπολογισμός.</p>

          <form id="propertyForm">
            <div class="grid">
              <div class="field">
                <label>Οδός</label>
                <input id="geo_street" value="Πατησίων" />
              </div>

              <div class="field">
                <label>Αριθμός</label>
                <input id="geo_number" value="1" />
              </div>

              <div class="field">
                <label>Πόλη</label>
                <input id="geo_city" value="Αθήνα" />
              </div>

              <div class="field">
                <label>Τ.Κ.</label>
                <input id="geo_postcode" value="10434" />
              </div>

              <div class="field">
                <label>Χώρα</label>
                <select id="geo_country">
                  <option selected>Ελλάδα</option>
                  <option>Κύπρος</option>
                  <option>Εξωτερικό</option>
                </select>
              </div>

              <div class="field">
                <label>Αξία Ακινήτου (€)</label>
                <input id="asset_value" type="number" value="100000" />
              </div>

              <div class="field">
                <label>Έτος Κατασκευής</label>
                <input id="year_construction" type="number" value="1990" />
              </div>

              <div class="field">
                <label>Αριθμός Ορόφων</label>
                <input id="floors_house" type="number" value="3" min="1" max="20" />
              </div>

              <div class="field">
                <label>Δείκτης Σταθερότητας</label>
                <select id="stability_number">
                  <option>1</option>
                  <option selected>3.1</option>
                  <option>3.2</option>
                </select>
              </div>

              <div class="field">
                <label>Υλικό Κατασκευής</label>
                <select id="material_house">
                  <option selected>Concrete</option>
                  <option>Wood</option>
                  <option>Metal</option>
                </select>
              </div>
            </div>

            <button type="submit">🚀 Υπολογισμός</button>
          </form>

          <div id="geoResult" class="result">Τα αποτελέσματα θα εμφανιστούν εδώ.</div>

          <div id="metrics" class="metrics" style="display:none;">
            <div class="metric"><h3>Σεισμός</h3><p id="eq"></p></div>
            <div class="metric"><h3>Φωτιά</h3><p id="fi"></p></div>
            <div class="metric"><h3>Πλημμύρα</h3><p id="fl"></p></div>
            <div class="metric"><h3>Άνεμος</h3><p id="wi"></p></div>
            <div class="metric"><h3>Χιόνι</h3><p id="sn"></p></div>
            <div class="metric"><h3>Κλοπή</h3><p id="th"></p></div>
          </div>

          <div id="totalBox" class="total" style="display:none;"></div>
        </div>
      </div>

      <script>
        const form = document.getElementById("propertyForm");
        const geoResult = document.getElementById("geoResult");
        const metrics = document.getElementById("metrics");
        const totalBox = document.getElementById("totalBox");

        form.addEventListener("submit", async function(e) {
          e.preventDefault();

          geoResult.textContent = "Γίνεται geocoding και υπολογισμός...";
          metrics.style.display = "none";
          totalBox.style.display = "none";

          const payload = {
            geo_street: document.getElementById("geo_street").value.trim(),
            geo_number: document.getElementById("geo_number").value.trim(),
            geo_city: document.getElementById("geo_city").value.trim(),
            geo_postcode: document.getElementById("geo_postcode").value.trim(),
            geo_country: document.getElementById("geo_country").value.trim(),
            asset_value: Number(document.getElementById("asset_value").value),
            year_construction: Number(document.getElementById("year_construction").value),
            floors_house: Number(document.getElementById("floors_house").value),
            stability_number: document.getElementById("stability_number").value,
            material_house: document.getElementById("material_house").value
          };

          try {
            const response = await fetch("/api/calculate", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload)
            });

            if (!response.ok) {
              const txt = await response.text();
              throw new Error(txt || "Server error");
            }

            const results = await response.json();

            let correctionsText = "-";
            if (results.corrections && results.corrections.length) {
              correctionsText = results.corrections.map(x => "- " + x).join("\\n");
            }

            geoResult.textContent =
`📍 Geocoding
Matched address: ${results.matched_address}
Longitude: ${Number(results.longitude).toFixed(6)}
Latitude: ${Number(results.latitude).toFixed(6)}
ΤΚ από shapefile: ${results.postcode_all}
2 πρώτα ψηφία ΤΚ: ${results.postcode_2digit}

🛠 Διορθώσεις / Παρατηρήσεις
${correctionsText}`;

            document.getElementById("eq").textContent = `${Number(results.earthquake.premium).toFixed(2)} €`;
            document.getElementById("fi").textContent = `${Number(results.fire.premium).toFixed(2)} €`;
            document.getElementById("fl").textContent = `${Number(results.flood.premium).toFixed(2)} €`;
            document.getElementById("wi").textContent = `${Number(results.wind.premium).toFixed(2)} €`;
            document.getElementById("sn").textContent = `${Number(results.snow.premium).toFixed(2)} €`;
            document.getElementById("th").textContent = `${Number(results.theft.premium).toFixed(2)} €`;

            metrics.style.display = "grid";
            totalBox.style.display = "block";
            totalBox.textContent = `💰 ΣΥΝΟΛΙΚΟ ΑΣΦΑΛΙΣΤΡΟ: ${Number(results.total.premium).toFixed(2)} € | Χρόνος: ${Number(results.total.seconds).toFixed(2)} sec`;
          } catch (err) {
            geoResult.textContent = "❌ Σφάλμα: " + err.message;
          }
        });
      </script>
    </body>
    </html>
    """


@app.post("/api/calculate")
def calculate(data: PropertyRequest):
    geo_res = geo_address_to_lonlat_with_report(
        geo_street=data.geo_street,
        geo_number=data.geo_number,
        geo_city=data.geo_city,
        geo_postcode=data.geo_postcode,
        geo_country=data.geo_country,
    )

    longitude = geo_res["longitude"]
    latitude = geo_res["latitude"]
    matched_address = geo_res["matched_address"]

    post_code_all = postcode_from_lonlat_int(longitude, latitude)
    post_code_2digits = int(str(post_code_all)[:2])

    results = run_all(
        lon=longitude,
        lat=latitude,
        asset_value=float(data.asset_value),
        year_construction=int(data.year_construction),
        floors_house=int(data.floors_house),
        stability_number=str(data.stability_number),
        material_house=str(data.material_house),
        post_code=post_code_2digits,
    )

    return {
        "matched_address": matched_address,
        "longitude": longitude,
        "latitude": latitude,
        "postcode_all": post_code_all,
        "postcode_2digit": post_code_2digits,
        "corrections": geo_res.get("corrections", []),
        **results
    }
