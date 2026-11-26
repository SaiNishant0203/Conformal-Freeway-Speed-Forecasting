"""Refresh the tables at the end of the audit report from results_audit/tables.md."""
p = "reports/Literature-driven audit.md"; s = open(p, encoding="utf-8").read()
i = s.index("<!-- TABLES -->") + len("<!-- TABLES -->")
open(p, "w", encoding="utf-8", newline="\n").write(s[:i] + "\n\n" + open("results_audit/tables.md", encoding="utf-8").read())
