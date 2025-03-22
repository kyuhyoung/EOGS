get = id => document.getElementById(id);

function author_node(author) {
    var span = document.createElement("span");
    var a = document.createElement("a");
    var sup = document.createElement("sup");
    a.textContent = author.name;
    // If the author email contains a "@" symbol, make it a mailto link
    if (author.email.includes("@"))
        a.href = "mailto:" + author.email;
    else
        a.href = author.email;
    sup.textContent = author.footnote.map(String).join(",");
    sup.textContent += author.affiliations.map(String).join(",");
    span.appendChild(a);
    span.appendChild(sup);
    return span
}

function affiliations_node(affiliations) {
    var span = document.createElement("span");
    span.innerHTML = affiliations.map((affiliation, index) =>
        "<sup>" + (index + 1).toString() + "</sup>" + affiliation
    ).join("<br>");
    return span
}

function footnote_node(footnotes) {
    var span = document.createElement("span");
    // footnotes is a list of pairs of the form [symbol, footnote]
    // Then make a string of the form "<sup>symbol</sup> footnote"
    // Then join the strings with ", "
    span.innerHTML = footnotes.map(footnote =>
        "<sup>" + footnote[0] + "</sup>" + footnote[1]
    ).join(", ");
    return span
}

function make_site(paper) {
    document.title = paper.title;
    get("title").textContent = paper.title;
    get("conference").textContent = paper.conference;

    // // Randomly swap the first two authors
    // if (Math.random() < 0.5) {
    //     var temp = paper.authors[0];
    //     paper.authors[0] = paper.authors[1];
    //     paper.authors[1] = temp;
    // }
    
    paper.authors.map((author, index) => {
        node = author_node(author);
        get("author-list").appendChild(node);
        if (index == paper.authors.length - 1) return;
        node.innerHTML += ", "
    })
    get("affiliation-list").appendChild(affiliations_node(paper.affiliations));
    get("footnote-list").appendChild(footnote_node(paper.footnotes));
    get("abstract").textContent = paper.abstract;

    // Populate the button list with the URLs from the paper
    buttonlist = get("button-list");
    for (var button in paper.URLs) {
        node = document.createElement("a");
        node.href = paper.URLs[button][0];

        img = document.createElement("img");
        img.src = paper.URLs[button][1];
        node.appendChild(img);

        span = document.createElement("span");
        span.textContent = button;
        node.appendChild(span);

        buttonlist.appendChild(node);
    }

    // Create the citation node at the end of the page in the bibtex div
    // and add a copy button to the bibtex div
    bibtex = get("bibtex");
    bibtextext = document.createElement("div");
    bibtextext.id = "bibtex-text";
    bibtextext.textContent = atob(paper.base64bibtex);
    var button = document.createElement("button");
    button.id = "copy-button";
    button.textContent = "Copy";
    button.onclick = () => {
        var range = document.createRange();
        range.selectNode(bibtextext);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(range);
        document.execCommand("copy");
        window.getSelection().removeAllRanges();
    }
    bibtex.appendChild(button);
    bibtex.appendChild(bibtextext);
}



fetch("./paper.json").then(response => response.json()).then(json => make_site(json));

// sliders = document.getElementsByClassName("slider-wrapper")
// for (var i = 0; i < sliders.length; i++) set_slider(sliders[i])

function set_single_comparer(rootfolder) {
    // In the root folder there should be 4 images:
    // gt.png, eogs.png, satngp.png, eonerf.png

    // Open all the images and append them to the <div id="comparisons"> element
    var comparisons = get("image-comparer-detailed-list");
    var image_comparer_detailed = document.createElement("div");
    image_comparer_detailed.className = "img-grid";
    comparisons.appendChild(image_comparer_detailed);

    var suffixes = ["", "_crop1", "_crop2"];
    var images = ["satngp", "eogs", "eonerf", "gt"];
    for (suffix of suffixes) {
        var image_horizontal_list = document.createElement("div");
        image_horizontal_list.className = "img-row";
        for (image of images) {
            var imgwrapper = document.createElement("div");
            imgwrapper.className = "img-wrapper";
            var img = document.createElement("img");
            img.src = rootfolder + "/" + image + suffix + ".png";
            if ((suffix === "_crop1") && (image === "gt")) {
                img.style = "outline: 10px solid magenta; outline-offset: -10px;"
            }
            if ((suffix === "_crop2") && (image === "gt")) {
                img.style = "outline: 10px solid cyan; outline-offset: -10px;"
            }

            imgwrapper.appendChild(img);
            var label = document.createElement("div");
            label.className = "image-label";
            if (image === "satngp") {
                label.textContent = "SAT-NGP";
            } else if (image === "eogs") {
                label.textContent = "EOGS";
            } else if (image === "eonerf") {
                label.textContent = "EO-NeRF";
            } else if (image === "gt") {
                label.textContent = "Ground Truth";
            } 
            imgwrapper.appendChild(label);
            image_horizontal_list.appendChild(imgwrapper);
        }
        image_comparer_detailed.appendChild(image_horizontal_list);
    }
}

set_single_comparer("assets/figures/visual_v2")
set_single_comparer("assets/figures_sup_v2/IARPA_001")
set_single_comparer("assets/figures_sup_v2/JAX_004")
set_single_comparer("assets/figures_sup_v2/JAX_068")
set_single_comparer("assets/figures_sup_v2/JAX_260")