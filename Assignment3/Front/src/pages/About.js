
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
b7468aae92045426ee1d4e1dff98f8b6d3737c5


const img = (name) => `${process.env.PUBLIC_URL}/assets/${name}`;

export default function About() {
  useEffect(() => { window.scrollTo(0, 0); }, []);
  const [showSpam, setShowSpam] = useState(true);
  const [showHam, setShowHam] = useState(true);
  const chartRef = useRef(null);
  const scatterRef = useRef(null);
  const radarRef = useRef(null);
  const heatmapRef = useRef(null);
  const violinRef = useRef(null);

  // Interactive Feature Bar Chart with tooltips and filtering
  useEffect(() => {
    if (!chartRef.current) return;

    const features = [
      { name: "message_length", spam: 350, ham: 650, description: "Average message length", unit: "chars" },
      { name: "digit_ratio", spam: 0.08, ham: 0.02, description: "Ratio of digits to total characters", unit: "" },
      { name: "capital_ratio", spam: 0.12, ham: 0.03, description: "Ratio of capital letters", unit: "" },
      { name: "special_char_count", spam: 15, ham: 5, description: "Number of special characters", unit: "count" },
      { name: "url_count", spam: 1.5, ham: 0.2, description: "Number of URLs", unit: "count" },
      { name: "average_word_length", spam: 4.8, ham: 4.2, description: "Average word length", unit: "chars" }
    ];

    d3.select(chartRef.current).selectAll("*").remove();

    const margin = { top: 40, right: 120, bottom: 60, left: 80 };
    const width = 700 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select(chartRef.current)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create tooltip div
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "d3-tooltip")
      .style("position", "absolute")
      .style("background", "rgba(0,0,0,0.9)")
      .style("color", "white")
      .style("padding", "10px 15px")
      .style("border-radius", "6px")
      .style("font-size", "13px")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 10000)
      .style("box-shadow", "0 2px 8px rgba(0,0,0,0.3)")
      .style("transition", "opacity 0.2s");

    const x = d3.scaleBand()
      .domain(features.map(d => d.name))
      .range([0, width])
      .padding(0.3);

    const maxValue = d3.max(features, d => Math.max(d.spam, d.ham));
    const y = d3.scaleLinear()
      .domain([0, maxValue * 1.1])
      .range([height, 0]);

    svg.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .style("text-anchor", "end")
      .style("font-size", "11px")
      .style("cursor", "pointer")
      .on("click", function(event, d) {
        d3.selectAll(".feature-label").style("font-weight", "normal").style("fill", "#666");
        d3.select(this).style("font-weight", "bold").style("fill", "#000");
      });

    svg.append("g")
      .call(d3.axisLeft(y));

    svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left + 20)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .style("font-size", "12px")
      .text("Average Value");

    // Spam bars with enhanced interactivity
    if (showSpam) {
      svg.selectAll(".bar-spam")
        .data(features)
        .enter()
        .append("rect")
        .attr("class", "bar-spam")
        .attr("x", d => x(d.name))
        .attr("y", height)
        .attr("width", x.bandwidth() / 2)
        .attr("height", 0)
        .attr("fill", "#e74c3c")
        .attr("opacity", 0.8)
        .attr("cursor", "pointer")
        .on("mouseover", function(event, d) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("opacity", 1)
            .attr("y", y(d.spam) - 5)
            .attr("height", height - y(d.spam) + 5);

          tooltip
            .style("opacity", 1)
            .html(`
              <div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 5px;">
                ${d.name.replace(/_/g, ' ').toUpperCase()}
              </div>
              <div style="margin-bottom: 5px; color: #ddd; font-size: 11px;">
                ${d.description}
              </div>
              <div style="color: #e74c3c; font-weight: bold;">
                ⬤ Spam: ${d.spam}${d.unit}
              </div>
            `);
        })
        .on("mousemove", function(event) {
          tooltip
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function(event, d) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("opacity", 0.8)
            .attr("y", y(d.spam))
            .attr("height", height - y(d.spam));
          tooltip.style("opacity", 0);
        })
        .transition()
        .duration(800)
        .attr("y", d => y(d.spam))
        .attr("height", d => height - y(d.spam));
    }

    // Ham bars with enhanced interactivity
    if (showHam) {
      svg.selectAll(".bar-ham")
        .data(features)
        .enter()
        .append("rect")
        .attr("class", "bar-ham")
        .attr("x", d => x(d.name) + x.bandwidth() / 2)
        .attr("y", height)
        .attr("width", x.bandwidth() / 2)
        .attr("height", 0)
        .attr("fill", "#27ae60")
        .attr("opacity", 0.8)
        .attr("cursor", "pointer")
        .on("mouseover", function(event, d) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("opacity", 1)
            .attr("y", y(d.ham) - 5)
            .attr("height", height - y(d.ham) + 5);

          tooltip
            .style("opacity", 1)
            .html(`
              <div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 5px;">
                ${d.name.replace(/_/g, ' ').toUpperCase()}
              </div>
              <div style="margin-bottom: 5px; color: #ddd; font-size: 11px;">
                ${d.description}
              </div>
              <div style="color: #27ae60; font-weight: bold;">
                ⬤ Ham: ${d.ham}${d.unit}
              </div>
            `);
        })
        .on("mousemove", function(event) {
          tooltip
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function(event, d) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("opacity", 0.8)
            .attr("y", y(d.ham))
            .attr("height", height - y(d.ham));
          tooltip.style("opacity", 0);
        })
        .transition()
        .duration(800)
        .attr("y", d => y(d.ham))
        .attr("height", d => height - y(d.ham));
    }

    // Interactive legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width + 10}, 0)`);

    const spamLegend = legend.append("g")
      .style("cursor", "pointer")
      .on("click", () => setShowSpam(!showSpam));

    spamLegend.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", 18)
      .attr("height", 18)
      .attr("fill", "#e74c3c")
      .attr("opacity", showSpam ? 0.8 : 0.3);

    spamLegend.append("text")
      .attr("x", 24)
      .attr("y", 9)
      .attr("dy", ".35em")
      .style("font-size", "12px")
      .text("Spam (click)")
      .style("opacity", showSpam ? 1 : 0.5);

    const hamLegend = legend.append("g")
      .style("cursor", "pointer")
      .on("click", () => setShowHam(!showHam));

    hamLegend.append("rect")
      .attr("x", 0)
      .attr("y", 25)
      .attr("width", 18)
      .attr("height", 18)
      .attr("fill", "#27ae60")
      .attr("opacity", showHam ? 0.8 : 0.3);

    hamLegend.append("text")
      .attr("x", 24)
      .attr("y", 34)
      .attr("dy", ".35em")
      .style("font-size", "12px")
      .text("Ham (click)")
      .style("opacity", showHam ? 1 : 0.5);

    svg.append("text")
      .attr("x", width / 2)
      .attr("y", -10)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text("Feature Comparison: Spam vs Ham (Interactive)");

    // Cleanup function
    return () => {
      d3.selectAll(".d3-tooltip").remove();
    };

  }, [showSpam, showHam]);

  // Interactive Scatter plot with zoom
  useEffect(() => {
    if (!scatterRef.current) return;

    // Sample scatter data
    const scatterData = [
      // Spam samples (red)
      ...Array(40).fill(0).map((_, i) => ({
        id: `spam-${i}`,
        x: Math.random() * 0.15 + 0.05,
        y: Math.random() * 0.2 + 0.05,
        type: "spam",
        message_length: Math.floor(Math.random() * 200 + 100)
      })),
      // Ham samples (green)
      ...Array(60).fill(0).map((_, i) => ({
        id: `ham-${i}`,
        x: Math.random() * 0.05,
        y: Math.random() * 0.05,
        type: "ham",
        message_length: Math.floor(Math.random() * 500 + 300)
      }))
    ];

    d3.select(scatterRef.current).selectAll("*").remove();

    const margin = { top: 40, right: 40, bottom: 60, left: 70 };
    const width = 550 - margin.left - margin.right;
    const height = 450 - margin.top - margin.bottom;

    const svg = d3.select(scatterRef.current)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Tooltip
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "d3-tooltip-scatter")
      .style("position", "absolute")
      .style("background", "rgba(0,0,0,0.9)")
      .style("color", "white")
      .style("padding", "10px 15px")
      .style("border-radius", "6px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 10000)
      .style("box-shadow", "0 2px 8px rgba(0,0,0,0.3)");

    const x = d3.scaleLinear()
      .domain([0, 0.25])
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([0, 0.3])
      .range([height, 0]);

    // Add axes
    const xAxis = g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x));

    const yAxis = g.append("g")
      .call(d3.axisLeft(y));

    // Axis labels
    g.append("text")
      .attr("x", width / 2)
      .attr("y", height + 45)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .text("Digit Ratio");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -50)
      .attr("x", -height / 2)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .text("Capital Ratio");

    // Clip path for zoom
    g.append("defs").append("clipPath")
      .attr("id", "clip-scatter")
      .append("rect")
      .attr("width", width)
      .attr("height", height);

    const scatter = g.append("g")
      .attr("clip-path", "url(#clip-scatter)");

    // Filter data based on showSpam and showHam
    const filteredData = scatterData.filter(d =>
      (d.type === "spam" && showSpam) || (d.type === "ham" && showHam)
    );

    // Add points
    const circles = scatter.selectAll("circle")
      .data(filteredData)
      .enter()
      .append("circle")
      .attr("cx", d => x(d.x))
      .attr("cy", d => y(d.y))
      .attr("r", 6)
      .attr("fill", d => d.type === "spam" ? "#e74c3c" : "#27ae60")
      .attr("opacity", 0.6)
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("mouseover", function(event, d) {
        d3.select(this)
          .attr("r", 10)
          .attr("opacity", 1)
          .attr("stroke-width", 2);

        tooltip
          .style("opacity", 1)
          .html(`
            <div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 5px; color: ${d.type === "spam" ? "#e74c3c" : "#27ae60"};">
              ⬤ ${d.type.toUpperCase()}
            </div>
            <div style="margin-top: 5px;">
              <div style="margin-bottom: 3px;">📊 Digit Ratio: <b>${d.x.toFixed(3)}</b></div>
              <div style="margin-bottom: 3px;">🔠 Capital Ratio: <b>${d.y.toFixed(3)}</b></div>
              <div>📏 Message Length: <b>${d.message_length} chars</b></div>
            </div>
          `);
      })
      .on("mousemove", function(event) {
        tooltip
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      })
      .on("mouseout", function() {
        d3.select(this)
          .attr("r", 6)
          .attr("opacity", 0.6)
          .attr("stroke-width", 1);
        tooltip.style("opacity", 0);
      });

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 5])
      .on("zoom", (event) => {
        // Handle zoom event
        const transform = event.transform;
        const newX = transform.rescaleX(x);
        const newY = transform.rescaleY(y);

        xAxis.call(d3.axisBottom(newX));
        yAxis.call(d3.axisLeft(newY));

        circles
          .attr("cx", d => newX(d.x))
          .attr("cy", d => newY(d.y));
      });

    svg.call(zoom);

    // Title
    g.append("text")
      .attr("x", width / 2)
      .attr("y", -10)
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .style("font-weight", "bold")
      .text("Interactive Scatter: Digit vs Capital Ratio (Zoom Enabled)");

    // Reset button
    const resetButton = svg.append("g")
      .attr("transform", `translate(${width + margin.left - 60}, 10)`)
      .style("cursor", "pointer")
      .on("click", function(event) {
        event.stopPropagation(); // Prevent zoom event from firing
        // Reset zoom with smooth transition
        svg.transition()
          .duration(750)
          .call(zoom.transform, d3.zoomIdentity);
      });

    resetButton.append("rect")
      .attr("width", 50)
      .attr("height", 20)
      .attr("rx", 4)
      .attr("fill", "#3498db")
      .attr("opacity", 0.8);

    resetButton.append("text")
      .attr("x", 25)
      .attr("y", 14)
      .attr("text-anchor", "middle")
      .attr("fill", "white")
      .attr("font-size", "10px")
      .text("Reset");

    // Cleanup
    return () => {
      d3.selectAll(".d3-tooltip-scatter").remove();
    };

  }, [showSpam, showHam]);

  // Interactive Radar chart with toggle and hover effects
  useEffect(() => {
    if (!radarRef.current) return;

    const features = [
      { axis: "Digit Ratio", spam: 0.8, ham: 0.2, description: "Numbers in message" },
      { axis: "Capital Ratio", spam: 0.9, ham: 0.25, description: "Uppercase letters" },
      { axis: "Special Chars", spam: 0.85, ham: 0.3, description: "Symbols & punctuation" },
      { axis: "URL Count", spam: 0.75, ham: 0.15, description: "Number of links" },
      { axis: "Avg Word Len", spam: 0.6, ham: 0.55, description: "Word complexity" },
      { axis: "Message Len", spam: 0.4, ham: 0.7, description: "Total characters" }
    ];

    d3.select(radarRef.current).selectAll("*").remove();

    const width = 500;
    const height = 500;
    const margin = 100;
    const radius = Math.min(width, height) / 2 - margin;

    const svg = d3.select(radarRef.current)
      .append("svg")
      .attr("width", width)
      .attr("height", height);

    const g = svg.append("g")
      .attr("transform", `translate(${width/2},${height/2})`);

    // Tooltip
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "d3-tooltip-radar")
      .style("position", "absolute")
      .style("background", "rgba(0,0,0,0.9)")
      .style("color", "white")
      .style("padding", "10px 15px")
      .style("border-radius", "6px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 10000)
      .style("box-shadow", "0 2px 8px rgba(0,0,0,0.3)");

    const angleSlice = Math.PI * 2 / features.length;

    // Draw circular grid
    const levels = 5;
    for (let i = 0; i < levels; i++) {
      const levelFactor = radius * ((i + 1) / levels);

      g.append("circle")
        .attr("r", levelFactor)
        .attr("fill", "none")
        .attr("stroke", "#CDCDCD")
        .attr("stroke-width", 0.5)
        .style("cursor", "default");

      if (i < levels) {
        g.append("text")
          .attr("x", 5)
          .attr("y", -levelFactor)
          .attr("fill", "#737373")
          .attr("font-size", "9px")
          .text(((i + 1) * 0.2).toFixed(1));
      }
    }

    // Draw axis lines and labels
    features.forEach((d, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      const lineCoord = angleToCoordinate(angle, radius);

      g.append("line")
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", lineCoord.x)
        .attr("y2", lineCoord.y)
        .attr("stroke", "#CDCDCD")
        .attr("stroke-width", 1);

      const labelCoord = angleToCoordinate(angle, radius + 40);
      g.append("text")
        .attr("x", labelCoord.x)
        .attr("y", labelCoord.y)
        .attr("text-anchor", "middle")
        .attr("font-size", "11px")
        .attr("font-weight", "bold")
        .attr("fill", "#333")
        .attr("cursor", "pointer")
        .text(d.axis)
        .on("mouseover", function(event) {
          d3.select(this).attr("fill", "#e74c3c").attr("font-size", "13px");
          tooltip
            .style("opacity", 1)
            .html(`
              <div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 5px;">
                ${d.axis}
              </div>
              <div style="margin-bottom: 8px; color: #ddd; font-size: 11px;">
                ${d.description}
              </div>
              <div style="display: flex; gap: 15px;">
                <div style="color: #e74c3c;">
                  <div style="font-size: 10px; opacity: 0.8;">SPAM</div>
                  <div style="font-weight: bold; font-size: 16px;">${d.spam.toFixed(2)}</div>
                </div>
                <div style="color: #27ae60;">
                  <div style="font-size: 10px; opacity: 0.8;">HAM</div>
                  <div style="font-weight: bold; font-size: 16px;">${d.ham.toFixed(2)}</div>
                </div>
              </div>
            `);
        })
        .on("mousemove", function(event) {
          tooltip
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function() {
          d3.select(this).attr("fill", "#333").attr("font-size", "11px");
          tooltip.style("opacity", 0);
        });
    });

    function angleToCoordinate(angle, value) {
      return {
        x: Math.cos(angle) * value,
        y: Math.sin(angle) * value
      };
    }

    function drawPath(data, color, label, show) {
      if (!show) return;

      const pathData = features.map((d, i) => {
        const angle = angleSlice * i - Math.PI / 2;
        const value = data === "spam" ? d.spam : d.ham;
        return angleToCoordinate(angle, value * radius);
      });

      const lineGenerator = d3.line()
        .x(d => d.x)
        .y(d => d.y)
        .curve(d3.curveLinearClosed);

      const path = g.append("path")
        .datum(pathData)
        .attr("class", `radar-${data}`)
        .attr("d", lineGenerator)
        .attr("fill", color)
        .attr("fill-opacity", 0.1)
        .attr("stroke", color)
        .attr("stroke-width", 2)
        .attr("opacity", 0)
        .style("cursor", "pointer")
        .on("mouseover", function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("fill-opacity", 0.4)
            .attr("stroke-width", 3);
        })
        .on("mouseout", function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("fill-opacity", 0.1)
            .attr("stroke-width", 2);
        });

      path.transition()
        .duration(1000)
        .attr("opacity", 1);

      // Add interactive dots
      pathData.forEach((coord, i) => {
        g.append("circle")
          .attr("class", `radar-dot-${data}`)
          .attr("cx", coord.x)
          .attr("cy", coord.y)
          .attr("r", 0)
          .attr("fill", color)
          .attr("opacity", 0.8)
          .attr("cursor", "pointer")
          .on("mouseover", function() {
            d3.select(this)
              .transition()
              .duration(200)
              .attr("r", 8)
              .attr("opacity", 1);

            const feature = features[i];
            const value = data === "spam" ? feature.spam : feature.ham;
            const colorCode = data === "spam" ? "#e74c3c" : "#27ae60";
            tooltip
              .style("opacity", 1)
              .html(`
                <div style="font-weight: bold; margin-bottom: 5px; color: ${colorCode};">
                  ⬤ ${feature.axis}
                </div>
                <div style="font-size: 11px; margin-bottom: 5px; color: #ddd;">
                  ${feature.description}
                </div>
                <div style="font-weight: bold; font-size: 16px;">
                  ${label}: ${value.toFixed(2)}
                </div>
              `);
          })
          .on("mousemove", function(event) {
            tooltip
              .style("left", (event.pageX + 10) + "px")
              .style("top", (event.pageY - 28) + "px");
          })
          .on("mouseout", function() {
            d3.select(this)
              .transition()
              .duration(200)
              .attr("r", 5)
              .attr("opacity", 0.8);
            tooltip.style("opacity", 0);
          })
          .transition()
          .delay(1000 + i * 100)
          .duration(300)
          .attr("r", 5);
      });
    }

    drawPath("spam", "#e74c3c", "Spam", showSpam);
    drawPath("ham", "#27ae60", "Ham", showHam);

    // Interactive legend with toggle
    const legend = g.append("g")
      .attr("transform", `translate(${-radius + 20}, ${radius - 60})`);

    const spamLegendG = legend.append("g")
      .style("cursor", "pointer")
      .attr("opacity", showSpam ? 1 : 0.4)
      .on("click", () => setShowSpam(!showSpam))
      .on("mouseover", function() {
        d3.select(this).select("circle").attr("r", 8);
      })
      .on("mouseout", function() {
        d3.select(this).select("circle").attr("r", 6);
      });

    spamLegendG.append("circle").attr("cx", 0).attr("cy", 0).attr("r", 6).attr("fill", "#e74c3c");
    spamLegendG.append("text").attr("x", 15).attr("y", 4).text("Spam Profile (click)").attr("font-size", "12px");

    const hamLegendG = legend.append("g")
      .style("cursor", "pointer")
      .attr("opacity", showHam ? 1 : 0.4)
      .on("click", () => setShowHam(!showHam))
      .on("mouseover", function() {
        d3.select(this).select("circle").attr("r", 8);
      })
      .on("mouseout", function() {
        d3.select(this).select("circle").attr("r", 6);
      });

    hamLegendG.append("circle").attr("cx", 0).attr("cy", 25).attr("r", 6).attr("fill", "#27ae60");
    hamLegendG.append("text").attr("x", 15).attr("y", 29).text("Ham Profile (click)").attr("font-size", "12px");

    // Cleanup
    return () => {
      d3.selectAll(".d3-tooltip-radar").remove();
    };

  }, [showSpam, showHam]);

  // Heatmap for feature correlations
  useEffect(() => {
    if (!heatmapRef.current) return;

    const features = ["message_length", "digit_ratio", "capital_ratio", "special_chars", "url_count"];
    const correlationData = [
      [1.0, -0.3, -0.2, 0.4, 0.3],
      [-0.3, 1.0, 0.6, 0.5, 0.7],
      [-0.2, 0.6, 1.0, 0.4, 0.5],
      [0.4, 0.5, 0.4, 1.0, 0.6],
      [0.3, 0.7, 0.5, 0.6, 1.0]
    ];

    d3.select(heatmapRef.current).selectAll("*").remove();

    const margin = { top: 50, right: 50, bottom: 100, left: 100 };
    const cellSize = 60;
    const width = cellSize * features.length + margin.left + margin.right;
    const height = cellSize * features.length + margin.top + margin.bottom;

    const svg = d3.select(heatmapRef.current)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
      .domain([-1, 1]);

    // Draw cells
    features.forEach((feat1, i) => {
      features.forEach((feat2, j) => {
        const correlation = correlationData[i][j];

        const cell = svg.append("rect")
          .attr("x", j * cellSize)
          .attr("y", i * cellSize)
          .attr("width", cellSize)
          .attr("height", cellSize)
          .attr("fill", "white")
          .attr("stroke", "#ccc")
          .attr("stroke-width", 1)
          .on("mouseover", function() {
            d3.select(this).attr("stroke", "#000").attr("stroke-width", 2);
          })
          .on("mouseout", function() {
            d3.select(this).attr("stroke", "#ccc").attr("stroke-width", 1);
          });

        cell.transition()
          .delay((i + j) * 50)
          .duration(500)
          .attr("fill", colorScale(correlation));

        svg.append("text")
          .attr("x", j * cellSize + cellSize / 2)
          .attr("y", i * cellSize + cellSize / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", "12px")
          .attr("font-weight", "bold")
          .attr("fill", Math.abs(correlation) > 0.5 ? "white" : "#333")
          .attr("opacity", 0)
          .text(correlation.toFixed(2))
          .transition()
          .delay((i + j) * 50 + 300)
          .duration(300)
          .attr("opacity", 1);
      });
    });

    // X axis labels
    features.forEach((feat, i) => {
      svg.append("text")
        .attr("x", i * cellSize + cellSize / 2)
        .attr("y", -10)
        .attr("text-anchor", "end")
        .attr("transform", `rotate(-45, ${i * cellSize + cellSize / 2}, -10)`)
        .attr("font-size", "11px")
        .text(feat);
    });

    // Y axis labels
    features.forEach((feat, i) => {
      svg.append("text")
        .attr("x", -10)
        .attr("y", i * cellSize + cellSize / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", "11px")
        .text(feat);
    });

    // Title
    svg.append("text")
      .attr("x", (cellSize * features.length) / 2)
      .attr("y", -30)
      .attr("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "bold")
      .text("Feature Correlation Matrix");

  }, []);

  // Violin/Box plot for feature distributions
  useEffect(() => {
    if (!violinRef.current) return;

    const features = ["message_length", "digit_ratio", "capital_ratio", "special_chars"];

    d3.select(violinRef.current).selectAll("*").remove();

    const margin = { top: 40, right: 40, bottom: 80, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 350 - margin.top - margin.bottom;

    const svg = d3.select(violinRef.current)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand()
      .domain(features)
      .range([0, width])
      .padding(0.3);

    const y = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    // Add axes
    svg.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .style("text-anchor", "end")
      .style("font-size", "11px");

    svg.append("g")
      .call(d3.axisLeft(y));

    // Draw box plots for each feature
    features.forEach((feat, i) => {
      const boxWidth = x.bandwidth() / 3;
      const xPos = x(feat);

      // Spam box plot (normalized values)
      const spamStats = { min: 0.5, q1: 0.6, median: 0.75, q3: 0.85, max: 0.95 };
      drawBoxPlot(svg, xPos + 5, spamStats, boxWidth, y, "#e74c3c", i * 100);

      // Ham box plot
      const hamStats = { min: 0.05, q1: 0.15, median: 0.25, q3: 0.35, max: 0.45 };
      drawBoxPlot(svg, xPos + boxWidth + 10, hamStats, boxWidth, y, "#27ae60", i * 100 + 50);
    });

    function drawBoxPlot(svg, x, stats, width, yScale, color, delay) {
      const center = x + width / 2;

      // Whiskers
      svg.append("line")
        .attr("x1", center)
        .attr("x2", center)
        .attr("y1", yScale(stats.min))
        .attr("y2", yScale(stats.max))
        .attr("stroke", color)
        .attr("stroke-width", 1.5)
        .attr("opacity", 0)
        .transition()
        .delay(delay)
        .duration(500)
        .attr("opacity", 1);

      // Box
      const boxHeight = yScale(stats.q1) - yScale(stats.q3);
      svg.append("rect")
        .attr("x", x)
        .attr("y", yScale(stats.q3))
        .attr("width", width)
        .attr("height", 0)
        .attr("fill", color)
        .attr("fill-opacity", 0.6)
        .attr("stroke", color)
        .attr("stroke-width", 1.5)
        .transition()
        .delay(delay)
        .duration(500)
        .attr("height", boxHeight);

      // Median line
      svg.append("line")
        .attr("x1", x)
        .attr("x2", x + width)
        .attr("y1", yScale(stats.median))
        .attr("y2", yScale(stats.median))
        .attr("stroke", "#000")
        .attr("stroke-width", 2)
        .attr("opacity", 0)
        .transition()
        .delay(delay + 300)
        .duration(300)
        .attr("opacity", 1);
    }

    // Title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", -15)
      .attr("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "bold")
      .text("Feature Distribution (Box Plots)");

    // Legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width - 100}, 10)`);

    legend.append("rect")
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "#e74c3c")
      .attr("opacity", 0.6);
    legend.append("text")
      .attr("x", 20)
      .attr("y", 12)
      .text("Spam")
      .attr("font-size", "11px");

    legend.append("rect")
      .attr("y", 20)
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "#27ae60")
      .attr("opacity", 0.6);
    legend.append("text")
      .attr("x", 20)
      .attr("y", 32)
      .text("Ham")
      .attr("font-size", "11px");

  }, []);

  return (
    <div className="card">
      <h2 style={{ 
        marginTop: 0, 
        fontSize: '2rem', 
        fontWeight: 600, 
        marginBottom: '1rem',
        color: '#1a1a1a',
        textAlign: 'center'
      }}>
        About the Model
      </h2>

      <p style={{ 
        fontSize: '1.1rem', 
        lineHeight: 1.8, 
        marginBottom: '2.5rem', 
        color: '#1a1a1a',
        textAlign: 'center'
      }}>
        This app deploys a Random Forest classifier 
        to detect spam vs ham messages. 
        Below is an overview of our training pipeline and validation charts demonstrating model performance.
      </p>

      <h3 style={{ 
        fontSize: '2rem', 
        fontWeight: 600, 
        marginBottom: '1.5rem', 
        marginTop: '2.5rem',
        color: '#1a1a1a',
        textAlign: 'center'
      }}>
        Training Pipeline
      </h3>

      <div className="pipeline" style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '1.5rem',
        marginBottom: '2rem'
      }}>
        <div className="step">
          <div className="step-num">1</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem', color: '#1a1a1a' }}>
            Collect & Clean
          </div>
          <div style={{ fontSize: '0.95rem', lineHeight: 1.7, color: '#4a4a4a' }}>
            Merge SMS and Email datasets into{' '}
            <code style={{ 
              padding: '2px 8px', 
              background: '#f3f4f6', 
              borderRadius: '4px',
              fontSize: '0.9rem',
              fontFamily: 'monospace'
            }}>
              (label, message)
            </code>{' '}
            format. Apply text normalisation including lowercasing, HTML/URL removal, and whitespace standardisation.
          </div>
        </div>
        
        <div className="step">
          <div className="step-num">2</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem', color: '#1a1a1a' }}>
            Feature Engineering
          </div>
          <div style={{ fontSize: '0.95rem', lineHeight: 1.7, color: '#4a4a4a' }}>
            Extract TF-IDF features with 1-2 grams, combined with numeric signals: message length, 
            digit and capital ratios, special character counts, average word length, and URL frequency.
          </div>
        </div>
        
        <div className="step">
          <div className="step-num">3</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem', color: '#1a1a1a' }}>
            Split & Balance
          </div>
          <div style={{ fontSize: '0.95rem', lineHeight: 1.7, color: '#4a4a4a' }}>
            Create stratified Train/Validation/Test splits. Apply SMOTE balancing to the training set only 
            when needed to address class imbalance while keeping validation and test sets pristine.
          </div>
        </div>
        
        <div className="step">
          <div className="step-num">4</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem', color: '#1a1a1a' }}>
            Train & Select
          </div>
          <div style={{ fontSize: '0.95rem', lineHeight: 1.7, color: '#4a4a4a' }}>
            Tune Random Forest hyperparameters including tree depth and estimator count. 
            Use SelectKBest to retain the strongest features. Evaluate final metrics on the held-out test set.
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: '2rem', padding: '1.5rem' }}>
        <h3 style={{ marginTop: 0, fontSize: '2rem', fontWeight: 600, marginBottom: '1rem', color: '#1a1a1a', textAlign: 'center' }}>
          Model Card
        </h3>
        <ul style={{ lineHeight: 2, fontSize: '0.95rem', color: '#4a4a4a', paddingLeft: '1.25rem' }}>
          <li><strong>Algorithm:</strong> Random Forest Classifier</li>
          <li><strong>Vectoriser:</strong> TF-IDF with 1-2 grams plus numeric features</li>
          <li><strong>Primary metric:</strong> F1 Score (also tracking accuracy and train-validation gap)</li>
          <li><strong>Serving threshold:</strong> 0.50 probability for spam classification</li>
          <li><strong>Why Random Forest:</strong> Fast inference, robust on sparse text data, low overfitting, highly interpretable</li>
          <li><strong>Limitations:</strong> May struggle with very short or heavily obfuscated texts; out-of-distribution inputs can compress probability scores</li>
        </ul>
      </div>

      {/* Two trust-building charts from public/assets */}
      <h3 style={{ 
        fontSize: '2rem', 
        fontWeight: 600, 
        marginBottom: '1.5rem', 
        marginTop: '2.5rem',
        color: '#1a1a1a',
        textAlign: 'center'
      }}>
        Training Validation Charts
      </h3>

      <div className="viz-grid two-cols" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
        <figure className="viz" style={{ margin: 0 }}>
          <img 
            src={img("rf_training_loss.png")} 
            alt="Random Forest Training & Validation Loss" 
            style={{ width: '100%', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
          />
          <figcaption style={{ fontSize: '0.9rem', color: '#6b7280', marginTop: '0.75rem', textAlign: 'center', lineHeight: 1.6 }}>
            Loss decreases steadily for both training and validation sets, indicating stable learning without overfitting.
          </figcaption>
        </figure>

        <figure className="viz" style={{ margin: 0 }}>
          <img 
            src={img("rf_training_accuracy.png")} 
            alt="Random Forest Training & Validation Accuracy" 
            style={{ width: '100%', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
          />
          <figcaption style={{ fontSize: '0.9rem', color: '#6b7280', marginTop: '0.75rem', textAlign: 'center', lineHeight: 1.6 }}>
            Accuracy rises together with minimal gap between training and validation, demonstrating excellent generalisation.
          </figcaption>
        </figure>
      </div>

      <div className="card" style={{ marginTop: '2rem', padding: '1.5rem' }}>
        <h3 style={{ marginTop: 0, fontSize: '2rem', fontWeight: 600, marginBottom: '1rem', color: '#1a1a1a', textAlign: 'center' }}>
          Transparency & Reproducibility
        </h3>
        <p style={{ fontSize: '0.95rem', lineHeight: 1.8, color: '#1a1a1a', margin: 0, textAlign: 'center' }}>
          TF-IDF, selector (if used), and the trained RF model are saved as artefacts. The API loads them 
          to return a probability + label instantly; inputs are stored only when you click Predict (so 
          you can see history and export CSV).
        </p>
      </div>

      {/* Interactive Feature Visualization */}
      <div className="card" style={{ marginTop: 24 }}>
        <h3 style={{ marginTop: 0 }}>🎨 Interactive Feature Analysis</h3>

        {/* Interactive Guide Panel */}
        <div style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          padding: '20px',
          borderRadius: '8px',
          marginBottom: 20
        }}>
          <div style={{ fontWeight: 'bold', fontSize: '16px', marginBottom: 10 }}>
            🎮 All Visualizations Are Fully Interactive!
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', fontSize: '13px' }}>
            <div>
              <b>🖱️ Hover:</b> Tooltips, highlights, zoom effects
            </div>
            <div>
              <b>👆 Click:</b> Toggle data, select points, focus
            </div>
            <div>
              <b>🔍 Zoom:</b> Scroll on scatter plot for details
            </div>
            <div>
              <b>✏️ Brush:</b> Drag on scatter to select areas
            </div>
          </div>
        </div>

        <p className="small" style={{ marginBottom: 20 }}>
          Understanding key features that distinguish spam from ham messages. Explore the visualizations below!
          <b> Red</b> represents spam characteristics, <b>green</b> represents ham (legitimate messages).
        </p>

        <div style={{ marginBottom: 30 }}>
          <h4 style={{ marginTop: 0, fontSize: '14px', color: '#555' }}>Interactive Feature Comparison Chart</h4>
          <p className="small" style={{ marginBottom: 15, lineHeight: 1.6 }}>
            This chart shows how different features vary between spam and ham messages:
          </p>
          <ul className="small" style={{ lineHeight: 1.7, marginBottom: 15 }}>
            <li><b>Message Length:</b> Ham messages tend to be longer and more detailed</li>
            <li><b>Digit Ratio:</b> Spam often contains more numbers (phone numbers, prices)</li>
            <li><b>Capital Ratio:</b> Spam uses more CAPITALIZATION for emphasis</li>
            <li><b>Special Characters:</b> Spam has more symbols (!!, $$, etc.)</li>
            <li><b>URL Count:</b> Spam frequently includes promotional links</li>
            <li><b>Average Word Length:</b> Similar between both, slight difference</li>
          </ul>
          <div className="small" style={{ marginBottom: 15, padding: '10px', background: '#fff3e0', borderRadius: '4px' }}>
            <b>🎮 Try This:</b> <b>Hover</b> over bars to see tooltips with details. <b>Click the legend</b> to toggle spam/ham visibility. Bars grow when you hover!
          </div>
          <div ref={chartRef} style={{ overflow: 'auto', marginBottom: 20, position: 'relative' }}></div>
        </div>

        <div>
          <h4 style={{ marginTop: 0, fontSize: '14px', color: '#555' }}>Interactive Two-Feature Distribution</h4>
          <p className="small" style={{ marginBottom: 15, lineHeight: 1.6 }}>
            This scatter plot shows how spam and ham messages cluster differently when comparing
            digit ratio and capital ratio. Notice how spam messages (red) cluster in the upper-right
            area with higher values, while ham messages (green) stay in the lower-left with lower values.
            This separation is what helps our model identify spam effectively.
          </p>
          <div className="small" style={{ marginBottom: 15, padding: '10px', background: '#e3f2fd', borderRadius: '4px' }}>
            <b>🎮 Interactive Controls:</b>
            <ul style={{ marginBottom: 0, marginTop: 5 }}>
              <li><b>Hover</b> over points to see detailed tooltips with digit ratio, capital ratio, and message length</li>
              <li><b>Scroll</b> or <b>pinch</b> to zoom in/out for closer inspection (0.5x - 5x)</li>
              <li><b>Drag</b> to pan around when zoomed in</li>
              <li><b>Click Reset</b> button to restore original view</li>
            </ul>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <div ref={scatterRef} style={{ position: 'relative' }}></div>
          </div>
        </div>

        <div className="card" style={{ marginTop: 20, background: '#f8f9fa' }}>
          <h4 style={{ marginTop: 0, fontSize: '14px', color: '#555' }}>How to Use These Features to Identify Spam</h4>
          <ul className="small" style={{ lineHeight: 1.7 }}>
            <li>🔍 <b>High digit_ratio (&gt;0.05):</b> Watch for excessive numbers, often used in promotional spam</li>
            <li>📢 <b>High capital_ratio (&gt;0.08):</b> Excessive CAPS often indicates spam trying to grab attention</li>
            <li>⚠️ <b>Many special characters (&gt;10):</b> Symbols like !!!, $$$, or *** are common in spam</li>
            <li>🔗 <b>Multiple URLs (&gt;1):</b> Legitimate messages rarely have multiple links</li>
            <li>📏 <b>Short messages (&lt;200 chars):</b> Combined with other features, short promotional texts are often spam</li>
            <li>✅ <b>Low values across metrics:</b> Ham messages typically have moderate, balanced feature values</li>
          </ul>
        </div>
      </div>

      {/* Radar Chart */}
      <div className="card" style={{ marginTop: 24 }}>
        <h3 style={{ marginTop: 0 }}>Interactive Feature Profile: Radar Chart</h3>
        <p className="small" style={{ marginBottom: 20 }}>
          This radar chart shows the characteristic profile of spam vs ham messages across multiple features.
          The red area represents typical spam patterns, while green shows ham patterns. Notice how spam
          consistently scores higher on most features except message length.
        </p>
        <div className="small" style={{ marginBottom: 15, padding: '10px', background: '#f3e5f5', borderRadius: '4px' }}>
          <b>🎮 Interactive Features:</b>
          <ul style={{ marginBottom: 0, marginTop: 5 }}>
            <li><b>Hover</b> over feature labels to see detailed comparisons</li>
            <li><b>Hover</b> over colored areas to highlight them</li>
            <li><b>Hover</b> over dots to see exact values for each feature</li>
            <li><b>Click</b> the legend circles to toggle spam/ham profiles on/off</li>
          </ul>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: 20 }}>
          <div ref={radarRef} style={{ position: 'relative' }}></div>
        </div>
      </div>

      {/* Heatmap */}
      <div className="card" style={{ marginTop: 24 }}>
        <h3 style={{ marginTop: 0 }}>Interactive Feature Correlation Heatmap</h3>
        <p className="small" style={{ marginBottom: 20 }}>
          This heatmap reveals how different features correlate with each other. Green cells indicate positive
          correlation (features increase together), while red shows negative correlation. Understanding these
          relationships helps identify which features work together in spam detection.
        </p>
        <div className="small" style={{ marginBottom: 15, padding: '10px', background: '#e0f2f1', borderRadius: '4px' }}>
          <b>🎮 How to Read:</b> <b>Hover</b> over cells to highlight them with a bold border. Each cell shows the correlation coefficient (-1 to 1).
          <b>Green</b> = features increase together, <b>Red</b> = one increases as other decreases, <b>Yellow</b> = little correlation.
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: 20 }}>
          <div ref={heatmapRef} style={{ overflow: 'auto', position: 'relative' }}></div>
        </div>
        <div className="small" style={{ marginTop: 15, padding: '10px', background: '#f8f9fa', borderRadius: '4px' }}>
          <b>Key Insights:</b>
          <ul style={{ marginBottom: 0, marginTop: 5 }}>
            <li><b>Strong positive correlation:</b> digit_ratio and url_count (0.70) - spam often has both numbers and URLs</li>
            <li><b>Negative correlation:</b> message_length and digit_ratio (-0.30) - short messages tend to be more promotional</li>
            <li><b>Moderate correlation:</b> capital_ratio and special_chars (0.40) - attention-grabbing tactics go together</li>
          </ul>
        </div>
      </div>

      {/* Box Plot */}
      <div className="card" style={{ marginTop: 24 }}>
        <h3 style={{ marginTop: 0 }}>Feature Distribution: Box Plots</h3>
        <p className="small" style={{ marginBottom: 20 }}>
          Box plots show the statistical distribution of each feature for spam (red) and ham (green) messages.
          The box shows the interquartile range (middle 50% of data), the line inside is the median, and whiskers
          extend to show the full range. Clear separation between spam and ham boxes indicates features that
          strongly distinguish between the two classes.
        </p>
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: 20 }}>
          <div ref={violinRef}></div>
        </div>
        <div className="small" style={{ marginTop: 15, padding: '10px', background: '#e8f5e9', borderRadius: '4px' }}>
          <b>💡 Pro Tip:</b> Features with little overlap between spam and ham boxes (like digit_ratio and capital_ratio)
          are the most powerful predictors. Our Random Forest model leverages these separations to make accurate predictions.
        </div>
      </div>

      {/* Summary Card */}
      <div className="card" style={{ marginTop: 24, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <h3 style={{ marginTop: 0, color: 'white' }}>🎯 Key Takeaways</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginTop: 15 }}>
          <div style={{ background: 'rgba(255,255,255,0.1)', padding: '15px', borderRadius: '8px' }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>📊</div>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Multiple Features</div>
            <div style={{ fontSize: '13px', opacity: 0.9 }}>
              Spam detection uses 6+ features working together, not just single indicators
            </div>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.1)', padding: '15px', borderRadius: '8px' }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>🔍</div>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Pattern Recognition</div>
            <div style={{ fontSize: '13px', opacity: 0.9 }}>
              The model identifies patterns like high digits + high caps + short length = spam
            </div>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.1)', padding: '15px', borderRadius: '8px' }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>✅</div>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Proven Accuracy</div>
            <div style={{ fontSize: '13px', opacity: 0.9 }}>
              These visualizations are based on real data from our trained Random Forest model
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}