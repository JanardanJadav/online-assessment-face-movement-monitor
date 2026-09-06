import { useState, useMemo } from "react";
import "./styles/index.css";
import CameraPanel from "./components/CameraPanel";

function App() {
    const sessionId = useMemo(() => {
    const now = new Date();

    const date = now.toISOString().slice(0, 10).replace(/-/g, "");
    const time = now.toTimeString().slice(0, 8).replace(/:/g, "");

    return `${date}-${time}`;
  }, []);

  const [events, setEvents] = useState([]);
  const [evidence, setEvidence] = useState([]);

  const [monitoringStatus, setMonitoringStatus] =
    useState("NOT STARTED");

  const [currentMovement, setCurrentMovement] =
    useState("LOOKING STRAIGHT");

  const [monitoringFaceCount, setMonitoringFaceCount] =
    useState(0);

  const exportCSV = () => {
    if (events.length === 0) return;

    const headers = ["Session ID","Date", "Timestamp", "Event"];

    const rows = events.map((event) => [
      event.sessionId,
      event.date,
      event.timestamp,
      event.type,
    ]);

    const csvContent = [
      headers,
      ...rows,
    ]
      .map((row) =>
        row
          .map((value) => `"${String(value).replace(/"/g, '""')}"`)
          .join(",")
      )
      .join("\n");

    const blob = new Blob([csvContent], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "face-movement-event-log.csv";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  return (
    <div className="app">

      {/* Header */}
      <header className="app-header">

        <div className="brand">
          <div className="brand-icon">
            FM
          </div>

          <div>
            <h1>Face Movement Monitor</h1>
            <p>Online Assessment Monitoring</p>
          </div>
        </div>
        <div className="session-info">
          <span className="session-label">
            SESSION
          </span>
          <span className="session-id">
            {sessionId}
          </span>
          <span className="status-dot" />
          <span className="live-text">
            SYSTEM READY
          </span>
        </div>
      </header>
      {/* Main Dashboard */}
      <main className="dashboard">

	{/* Camera Section */}
	<CameraPanel
        setEvents={setEvents}
        setEvidence={setEvidence}
        sessionId={sessionId}
        setMonitoringStatus={setMonitoringStatus}
        setCurrentMovement={setCurrentMovement}
        setMonitoringFaceCount={setMonitoringFaceCount}
  />
        {/* Monitoring Panel */}
        <section className="monitor-card">
          <div className="card-header">
            <div>
              <h2>Monitoring Status</h2>
              <p>
                Current assessment state
              </p>
            </div>

            <span
              className={`monitor-badge ${
                monitoringStatus === "NOT STARTED"
                  ? "monitor-not-started"
                  : monitoringStatus === "STARTED"
                    ? "monitor-started"
                    : "monitor-active"
              }`}
            >
              {monitoringStatus}
            </span>

          </div>

          <div className="movement-panel">
            <span className="movement-label">
              CURRENT MOVEMENT
            </span>
            <h2>
              {currentMovement}
            </h2>
            <p>
              {monitoringStatus === "NOT STARTED"
              ? "Waiting for camera"
              : monitoringStatus === "STARTED"
              ? "Waiting for calibration"
              : "Monitoring active"}
            </p>
          </div>

          <div className="face-info">

            <div className="info-item">
              <span>
                FACE COUNT
              </span>

              <strong>
                {monitoringFaceCount}
              </strong>

            </div>

            <div className="info-item">

              <span>
                MONITORING
              </span>

              <strong>
                {monitoringStatus === "ACTIVE" ? "ON" : "OFF"}
              </strong>

            </div>

          </div>

        </section>

        {/* Statistics */}
        <section className="stats-section">
          <div className="section-heading">
            <div>
              <h2>Event Overview</h2>
              <p>
                Session activity summary
              </p>
            </div>

            <span className="event-count">
              {events.length} Events
            </span>

          </div>


          <div className="stats-grid">

            <div className="stat-card">
              <span>LOOKING LEFT</span>
              <strong>
                {events.filter((event) => event.type === "LOOKING LEFT").length}
              </strong>
            </div>

            <div className="stat-card">
              <span>LOOKING RIGHT</span>
              <strong>
                {events.filter((event) => event.type === "LOOKING RIGHT").length}
              </strong>
            </div>

            <div className="stat-card">
              <span>LOOKING UP</span>
              <strong>
                {events.filter((event) => event.type === "LOOKING UP").length}
              </strong>
            </div>

            <div className="stat-card">
              <span>LOOKING DOWN</span>
              <strong>
                {events.filter((event) => event.type === "LOOKING DOWN").length}
              </strong>
            </div>

            <div className="stat-card">
              <span>NO FACE</span>
              <strong>
                {events.filter((event) => event.type === "NO FACE").length}
              </strong>
            </div>
            <div className="stat-card">
              <span>MULTIPLE FACES</span>
              <strong>
                {events.filter((event) => event.type === "MULTIPLE FACES").length}
              </strong>
            </div>
          </div>
        </section>

{/* Evidence */}
<section className="content-card">

  <div className="section-heading">

    <div>
      <h2>Evidence</h2>

      <p>
        Captured monitoring events
      </p>
    </div>

    <span className="event-count">
      {evidence.length} Captures
    </span>

  </div>

  {evidence.length === 0 ? (
    <div className="empty-state">

      <div className="empty-icon">
        ◫
      </div>

      <h3>
        No evidence captured
      </h3>

      <p>
        Evidence will appear here when a
        monitoring event is detected.
      </p>

    </div>
  ) : (
    <div className="evidence-gallery">

      {evidence.map((item) => (
        <div className="evidence-item" key={item.id}>

          <img
            src={item.image}
            alt={`${item.type} evidence`}
            style={{
              width: "100%",
              maxWidth: "720px",
              height: "auto",
              objectFit: "cover",
              borderRadius: "8px",
            }}
          />

        </div>
      ))}

    </div>
  )}

</section>

        {/* Event Log */}
        <section className="content-card">

          <div className="section-heading">

            <div>
              <h2>Event Log</h2>

              <p>
                Session monitoring history
              </p>
            </div>

            <button
              className="btn export-btn"
              onClick={exportCSV}
              disabled={events.length === 0}
            >
              Export CSV
            </button>

          </div>

          {events.length === 0 ? (
            <div className="empty-state">

              <div className="empty-icon">
                ≡
              </div>

              <h3>
                No events recorded
              </h3>

              <p>
                Evidence will appear here when a
                monitoring event is detected.
              </p>

            </div>
          ) : (
            <div className="event-log">

            <div className="event-log-header">
              <span>SESSION ID</span>
              <span>DATE</span>
              <span>TIME</span>
              <span>EVENT</span>
            </div>

            {events.map((event) => (
              <div className="event-row" key={event.id}>

                <span className="event-session-id">
                  {event.sessionId}
                </span>

                <span className="event-date">
                  {event.date}
                </span>

                <span className="event-time">
                  {event.timestamp}
                </span>

                <span className={`event-type event-${event.type
                  .toLowerCase()
                  .replace(/\s+/g, "-")}`}>
                  {event.type}
                </span>

              </div>
            ))}

          </div>
          )}

        </section>

      </main>

    </div>
  );
}

export default App;