-- 1. first level-1 heading -> document title; if it was the only one, promote the rest one level
-- 2. <div style="color:#RRGGBB"> -> colored LaTeX group (answers are green)
-- 3. every code block -> fvextra Verbatim with line breaking (long lines wrap instead of overflowing)
function Pandoc(doc)
  local h1, first = 0, nil
  for i, b in ipairs(doc.blocks) do
    if b.t == "Header" and b.level == 1 then h1 = h1 + 1; first = first or i end
  end
  -- drop a hand-written "**Contents**" list (LaTeX makes a real table of contents with page numbers)
  for i, b in ipairs(doc.blocks) do
    if b.t == "Para" and pandoc.utils.stringify(b) == "Contents" and doc.blocks[i + 1] and doc.blocks[i + 1].t == "BulletList" then
      doc.blocks:remove(i + 1); doc.blocks:remove(i); break
    end
  end
  if first and not doc.meta.title then
    doc.meta.title = doc.blocks[first].content
    doc.blocks:remove(first)
    if h1 == 1 then
      doc.blocks = doc.blocks:walk({ Header = function(h) h.level = math.max(1, h.level - 1); return h end })
    end
  end
  return doc
end
function Div(d)
  local c = (d.attributes.style or ""):match("color:%s*#(%x%x%x%x%x%x)")
  if not c then return nil end
  local out = { pandoc.RawBlock("latex", "\\begingroup\\color[HTML]{" .. c:upper() .. "}") }
  for _, b in ipairs(d.content) do table.insert(out, b) end
  table.insert(out, pandoc.RawBlock("latex", "\\endgroup"))
  return out
end
function CodeBlock(cb)
  return pandoc.RawBlock("latex", "\\begin{Verbatim}[breaklines,breakanywhere,fontsize=\\small,frame=leftline,framerule=0.6pt,rulecolor=\\color{black!25},xleftmargin=4pt]\n"
    .. cb.text .. "\n\\end{Verbatim}")
end

-- subscript letters missing from every installed font (e.g. ωₖ in the RoPE notes) -> \textsubscript{k}
local SUBS = { ["ₖ"] = "k", ["ₙ"] = "n", ["ₘ"] = "m", ["ₜ"] = "t" }
local function Str(el)
  local out, buf, hit = {}, "", false
  for _, c in utf8.codes(el.text) do
    local ch = utf8.char(c)
    if SUBS[ch] then
      hit = true
      if buf ~= "" then table.insert(out, pandoc.Str(buf)); buf = "" end
      table.insert(out, pandoc.RawInline("latex", "\\textsubscript{" .. SUBS[ch] .. "}"))
    else buf = buf .. ch end
  end
  if not hit then return nil end
  if buf ~= "" then table.insert(out, pandoc.Str(buf)) end
  return out
end

-- "**Label:** ![img](...)" in one paragraph -> the label, then the image centered on its own line
local function Para(el)
  local imgs, rest = {}, {}
  for _, x in ipairs(el.content) do
    if x.t == "Image" then table.insert(imgs, x) else table.insert(rest, x) end
  end
  if #imgs == 0 or #rest == 0 then return nil end
  while #rest > 0 and (rest[#rest].t == "Space" or rest[#rest].t == "SoftBreak") do table.remove(rest) end
  -- a minipage keeps the label and its figure on the same page
  local out = { pandoc.RawBlock("latex", "\\par\\noindent\\begin{minipage}{\\linewidth}"), pandoc.Para(rest),
                pandoc.RawBlock("latex", "\\begin{center}") }
  for _, im in ipairs(imgs) do table.insert(out, pandoc.Plain({ im })) end
  table.insert(out, pandoc.RawBlock("latex", "\\end{center}\\end{minipage}\\par\\medskip"))
  return out
end


-- Table column widths from content (like a browser's auto layout), estimated in pt:
--   natural fit -> no wrapping; otherwise every column gets its longest unbreakable word and the rest of
--   the line is shared in proportion to how much each column would still like; if even the minimums don't
--   fit, retry at \small then \footnotesize.
local LINEWIDTH, TABCOLSEP = 487, 6          -- pt, for 11pt article with 2.2cm margins on Letter paper

local function inline_widths(inls, size, mono, bold, acc)
  -- acc.cur = width of the word being built, acc.words = finished word widths, acc.total = whole text
  local cw = size * (mono and 0.525 or (bold and 0.57 or 0.5))
  for _, x in ipairs(inls) do
    if x.t == "Str" then
      for _ in utf8.codes(x.text) do acc.cur = acc.cur + cw; acc.total = acc.total + cw end
    elseif x.t == "Code" then
      for _, c in utf8.codes(x.text) do
        if c == 32 then table.insert(acc.words, acc.cur); acc.cur = 0
        else acc.cur = acc.cur + size * 0.525 end
        acc.total = acc.total + size * 0.525
      end
    elseif x.t == "Math" then
      local w = #x.text * size * 0.33          -- TeX source is longer than what it typesets
      acc.cur = acc.cur + w; acc.total = acc.total + w
    elseif x.t == "Space" or x.t == "SoftBreak" or x.t == "LineBreak" then
      table.insert(acc.words, acc.cur); acc.cur = 0; acc.total = acc.total + size * 0.33
    elseif x.content then
      inline_widths(x.content, size, mono, bold or x.t == "Strong", acc)
    end
  end
end

local function cell_size(cell, size)
  local acc = { cur = 0, words = {}, total = 0 }
  for _, b in ipairs(cell.contents) do
    if b.content then inline_widths(b.content, size, false, false, acc) end
    table.insert(acc.words, acc.cur); acc.cur = 0
  end
  local longest = 0
  for _, w in ipairs(acc.words) do longest = math.max(longest, w) end
  return longest, acc.total
end

local function layout(tbl, size)
  local n = #tbl.colspecs
  local mins, prefs = {}, {}
  for i = 1, n do mins[i], prefs[i] = 0, 0 end
  local function scan(rows, is_head)
    for _, row in ipairs(rows) do
      for i, cell in ipairs(row.cells) do
        if i <= n then
          local lo, total = cell_size(cell, size)
          mins[i] = math.max(mins[i], lo)
          if not is_head then prefs[i] = math.max(prefs[i], total) end   -- headers may wrap
        end
      end
    end
  end
  scan(tbl.head.rows, true)
  for _, body in ipairs(tbl.bodies) do scan(body.body, false) end
  local avail = LINEWIDTH - 2 * TABCOLSEP * n - 4
  local smin, spref = 0, 0
  for i = 1, n do prefs[i] = math.max(prefs[i], mins[i]); smin = smin + mins[i]; spref = spref + prefs[i] end
  if spref <= avail then return "natural" end
  if smin > avail then return nil end
  local widths, extra = {}, (avail - smin) / math.max(spref - smin, 1e-9)
  for i = 1, n do widths[i] = (mins[i] + (prefs[i] - mins[i]) * extra) / (LINEWIDTH - 2 * TABCOLSEP * (n - 1)) end
  return widths
end

local function Table(tbl)
  -- natural width at 11pt, then natural at \small; otherwise wrap at \small, then at \footnotesize
  for _, sz in ipairs({ { 11, nil, true }, { 10, "\\small", true }, { 10, "\\small" }, { 9, "\\footnotesize" } }) do
    local w = layout(tbl, sz[1])
    if w and (w == "natural" or not sz[3]) then
      for i, cs in ipairs(tbl.colspecs) do
        tbl.colspecs[i] = { cs[1], (w ~= "natural") and w[i] or nil }
      end
      if not sz[2] then return tbl end
      return { pandoc.RawBlock("latex", "\\begingroup" .. sz[2]), tbl, pandoc.RawBlock("latex", "\\endgroup") }
    end
  end
  return nil
end

return { { Pandoc = Pandoc }, { Div = Div, CodeBlock = CodeBlock, Str = Str, Para = Para, Table = Table } }
