-- Populate author, affiliation, correspondence, ORCID, and keywords for DOCX.
-- Pandoc's DOCX writer reads only `meta.author`, ignoring Quarto's structured metadata.

local str = pandoc.utils.stringify

local function md_inlines(text)
  return pandoc.MetaInlines(pandoc.read(text, "markdown").blocks[1].content)
end

function Pandoc(doc)
  if not quarto.doc.is_format("docx") then
    return nil
  end

  local authors = doc.meta["by-author"] or {}
  if #authors > 0 then
    local names, emails, orcids = {}, {}, {}
    for _, a in ipairs(authors) do
      local name = str(a.name.literal or a.name)
      local marks = {}
      for _, aff in ipairs(a.affiliations or {}) do
        marks[#marks + 1] = str(aff.number)
      end
      if a.attributes and a.attributes.corresponding then
        marks[#marks + 1] = "\\*"
        if a.email then
          local e = str(a.email)
          emails[#emails + 1] = string.format("[%s](mailto:%s)", e, e)
        end
      end
      names[#names + 1] = #marks > 0
        and string.format("%s^%s^", name, table.concat(marks, ","))
        or name
      if a.orcid then
        local id = str(a.orcid)
        orcids[#orcids + 1] = string.format("%s ([%s](https://orcid.org/%s))", name, id, id)
      end
    end

    local lines = { md_inlines(table.concat(names, ", ")) }
    for _, aff in ipairs(doc.meta["by-affiliation"] or {}) do
      lines[#lines + 1] = md_inlines(string.format("^%s^ %s", str(aff.number), str(aff.name)))
    end
    if #emails > 0 then
      lines[#lines + 1] = md_inlines("^\\*^ Correspondence: " .. table.concat(emails, ", "))
    end
    if #orcids > 0 then
      lines[#lines + 1] = md_inlines("ORCID: " .. table.concat(orcids, "; "))
    end
    doc.meta.author = pandoc.MetaList(lines)
  end

  local kw = doc.meta.keywords or {}
  if #kw > 0 then
    local items = {}
    for _, k in ipairs(kw) do
      items[#items + 1] = str(k)
    end
    table.insert(
      doc.blocks,
      1,
      pandoc.read("**Keywords:** " .. table.concat(items, "; "), "markdown").blocks[1]
    )
  end

  return doc
end
