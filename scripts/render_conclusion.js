// Pure docx layout: reads a fully-prepared JSON payload and renders the
// Заключение structure. All data wrangling (filtering, grouping, number
// formatting, Russian text composition) happens on the Python side in
// core/conclusion_generator.py — this script only lays out what it's given.
//
// Invoked as: node render_conclusion.js <payload.json> <output.docx>

const fs = require("fs");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  WidthType,
  ShadingType,
  AlignmentType,
  VerticalAlign,
} = require("docx");

const [, , payloadPath, outputPath] = process.argv;
if (!payloadPath || !outputPath) {
  console.error("usage: node render_conclusion.js <payload.json> <output.docx>");
  process.exit(1);
}

const payload = JSON.parse(fs.readFileSync(payloadPath, "utf-8"));

const NAVY = "1F3A5F";
const WHITE = "FFFFFF";
// Letter portrait (docx.js default: 12240 DXA wide) minus 1in (1440 DXA) margins on each side.
const TABLE_WIDTH = 9360;
const ITEM_TABLE_WIDTHS = [1400, 3000, 1900, 3060];
const SUMMARY_TABLE_WIDTHS = [4000, 2680, 2680];

function headerCell(text, width) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, color: "auto", fill: NAVY },
    verticalAlign: VerticalAlign.CENTER,
    children: [
      new Paragraph({
        children: [new TextRun({ text, bold: true, color: WHITE })],
      }),
    ],
  });
}

function bodyCell(text, width, { bold = false } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    children: [
      new Paragraph({
        children: [new TextRun({ text: text == null ? "" : String(text), bold })],
      }),
    ],
  });
}

function dataTable(headers, widths, rows, { boldRows = [] } = {}) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => headerCell(h, widths[i])),
  });
  const bodyRows = rows.map((cells, rowIndex) =>
    new TableRow({
      children: cells.map((cell, i) =>
        bodyCell(cell, widths[i], { bold: boldRows.includes(rowIndex) })
      ),
    })
  );
  return new Table({
    width: { size: TABLE_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: [headerRow, ...bodyRows],
  });
}

function heading(text) {
  return new Paragraph({
    children: [new TextRun({ text, bold: true })],
    spacing: { before: 300, after: 150 },
  });
}

function paragraph(text, { italics = false, bold = false, alignment } = {}) {
  return new Paragraph({
    alignment,
    children: [new TextRun({ text, italics, bold })],
    spacing: { after: 150 },
  });
}

const children = [];

// --- Header ---------------------------------------------------------------
children.push(
  paragraph("ЗАКЛЮЧЕНИЕ ПО РЕЗУЛЬТАТАМ ЭКСПЕРТИЗЫ", {
    bold: true,
    alignment: AlignmentType.CENTER,
  })
);
children.push(
  paragraph(`проекта бюджета ИТ-расходов ${payload.entity} на ${payload.budget_year} год`, {
    bold: true,
    alignment: AlignmentType.CENTER,
  })
);
children.push(new Paragraph({ text: "" }));
children.push(paragraph(payload.intro_text));
if (payload.executive_summary) {
  children.push(paragraph(payload.executive_summary, { italics: true }));
}

// --- Section 1: АТК ---------------------------------------------------------
children.push(heading("1. Статьи, выносимые на рассмотрение АТК"));
children.push(paragraph(payload.atk_section.intro));
children.push(
  dataTable(
    ["Вкладка", "Статья / контрагент", `Сумма ${payload.budget_year}, тыс. тг`, "Основание для вынесения на АТК"],
    ITEM_TABLE_WIDTHS,
    payload.atk_section.rows.map((r) => [r.tab, r.item, r.amount, r.reason])
  )
);
children.push(paragraph(payload.atk_section.footer, { bold: true }));

// --- Section 2: уточнение ----------------------------------------------------
children.push(heading(`2. Статьи, требующие уточнения от ${payload.entity}`));
children.push(
  dataTable(
    ["Вкладка", "Статья / контрагент", `Сумма ${payload.budget_year}, тыс. тг`, "Суть уточнения"],
    ITEM_TABLE_WIDTHS,
    payload.clarification_section.rows.map((r) => [r.tab, r.item, r.amount, r.reason])
  )
);
children.push(paragraph(payload.clarification_section.footer, { bold: true }));

// --- Section 3: согласовано ---------------------------------------------------
children.push(heading("3. Статьи, согласованные без замечаний"));
children.push(paragraph(payload.approved_section.intro));
for (const group of payload.approved_section.groups) {
  const items = group.items.length ? group.items.join(", ") : "—";
  children.push(paragraph(`${group.label}: ${items}.`));
}
children.push(paragraph(payload.approved_section.footer, { bold: true }));

// --- Резюме -------------------------------------------------------------------
children.push(heading("Резюме"));
const summaryRows = payload.summary_table.rows;
children.push(
  dataTable(
    ["Решение", "Количество статей", "Сумма, тыс. тг"],
    SUMMARY_TABLE_WIDTHS,
    summaryRows.map((r) => [r.decision, r.count, r.amount]),
    { boldRows: summaryRows.map((r, i) => (r.bold ? i : -1)).filter((i) => i >= 0) }
  )
);

const doc = new Document({ sections: [{ properties: {}, children }] });

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
});
