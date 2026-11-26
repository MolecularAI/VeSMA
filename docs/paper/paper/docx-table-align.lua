-- Work around Quarto 1.10 / Pandoc DOCX nested-table alignment leak:
-- Quarto wraps captioned tables in a 1-column centered layout table, and Pandoc's DOCX
-- writer leaks that outer AlignCenter as a trailing `<w:jc w:val="center"/>` on every
-- paragraph inside the nested table (overriding each column's real alignment in Word).
-- Resetting the outer wrapper column/cell to AlignDefault preserves the inner table's alignments.

local function has_nested_table(tbl)
  local found = false
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.body) do
      for _, cell in ipairs(row.cells) do
        for _, block in ipairs(cell.contents) do
          if block.t == "Table" then
            return true
          end
          pandoc.walk_block(block, {
            Table = function(_)
              found = true
            end,
          })
          if found then
            return true
          end
        end
      end
    end
  end
  return false
end

function Table(tbl)
  if not quarto.doc.is_format("docx") then
    return nil
  end
  if #tbl.colspecs == 1 and has_nested_table(tbl) then
    tbl.colspecs[1][1] = pandoc.AlignDefault
    for _, body in ipairs(tbl.bodies) do
      for _, row in ipairs(body.body) do
        for _, cell in ipairs(row.cells) do
          cell.alignment = pandoc.AlignDefault
        end
      end
    end
    return tbl
  end
  return nil
end
