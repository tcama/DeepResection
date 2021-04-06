DROP TABLE IF EXISTS feedback;

CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_time DATE NOT NULL,
    rating INTEGER NOT NULL,
    comments TEXT
);
