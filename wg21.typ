#let make-venue(paper_number, audience) = move(dy: -1.9cm, {
  box(rect(fill: luma(140), inset: 10pt, height: 2.5cm)[
    #set text(font: "TeX Gyre Pagella", fill: white, weight: 700, size: 20pt)
    #align(bottom)[ISO/IEC JTC1/SC22/WG21]
  ])
  set text(22pt, font: "TeX Gyre Heros")
    box(width: 1fr, pad(left: 10pt, bottom: 10pt, align(right, paper_number)))
  set text(10pt, font: "TeX Gyre Heros")
  v(0pt)
  [Audience: #audience]
})

#let make-title(
  title,
  authors,
  abstract,
  date,
) = {
  set par(spacing: 1em)
  set text(font: "TeX Gyre Heros")

  par(
    justify: false,
    text(24pt, fill: rgb("004b71"), title, weight: "bold")
  )

  text(12pt,
    authors.enumerate()
    .map(((i, author)) => box[#author.name #super[#(i+1)]])
    .join(", ")
  )
  parbreak()

  for (i, author) in authors.enumerate() [
    #set text(8pt)
    #super[#(i+1)]
    #author.institution
    #link("mailto:" + author.mail) \
  ]
  v(8pt)
  [#date.month #date.day, #date.year]
  v(8pt)
  set text(10pt)
  set par(justify: true)

  [
    #heading(outlined: false, bookmarked: false)[Abstract]
    #text(font: "TeX Gyre Pagella", abstract)
  ]
  v(18pt)
}

#let template(
    paper_number: "D????R?",
    audience: "LEWG",
    title: [],
    authors: (),
    date: [],
    doi: "",
    keywords: (),
    abstract: [],
    make-venue: make-venue,
    make-title: make-title,
    body,
) = {
    set page(
      paper: "us-letter",
      margin: (top: 1.9cm, bottom: 1in, x: 1.6cm),
      columns: 2
    )
    set par(justify: true)
    set text(10pt, font: "TeX Gyre Pagella")
    set list(indent: 8pt)
    // show link: set text(underline: false)
    show heading: set text(size: 11pt)
    show heading.where(level: 1): set text(font: "TeX Gyre Heros", fill: rgb("004b71"), size: 12pt)
    show heading: set block(below: 8pt)
    show heading.where(level: 1): set block(below: 12pt)
    let bad(body) = highlight(
        fill: rgb("#CC0000"),
        extent: 2pt,
        top-edge: 8pt,
        radius: 2pt,
        {
          set text(white)
          strong(body)
        }
    )
    let good(body) = highlight(
        fill: rgb("#00CC00"),
        extent: 2pt,
        top-edge: 8pt,
        radius: 2pt,
        {
          set text(white)
          strong(body)
        }
    )

    show raw.where(lang: "Cpp"): r => {
      show "Undefined behavior": bad
      show "Okay": good
      r
    }

    // Don't line break in the middle of "C++" in prose
    show "C++": box("C++")

    place(make-venue(paper_number, audience), top, scope: "parent", float: true)
    place(
      make-title(title, authors, abstract, date),
      top,
      scope: "parent",
      float: true
    )

    show figure: align.with(center)
    show figure: set text(8pt)
    show figure.caption: pad.with(x: 10%)

    set heading(numbering: "1.")
    // set raw(theme: none)
    body
  }

#let standardese(body) = {
  block(
      inset: 1em,
      stroke: (left: 1pt + black)
  )[
      #set list(marker: [--])
      #body
  ]
}
