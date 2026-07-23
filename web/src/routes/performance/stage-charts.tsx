import * as React from "react";
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";
import { PIPELINE_STAGES, PIPELINE_TOTAL_MS, STAGE_RAMP } from "@/config/pipeline";
import { formatDuration, formatPercent } from "@/lib/format";
import { useResolvedTheme } from "@/hooks/use-theme";

/** Ink colours read from the active theme so charts match the surrounding UI. */
function useChartInk() {
  const theme = useResolvedTheme();
  return React.useMemo(
    () =>
      theme === "dark"
        ? { muted: "#93a2bd", grid: "#1e2a41", ramp: STAGE_RAMP.dark }
        : { muted: "#55627a", grid: "#dfe4ec", ramp: STAGE_RAMP.light },
    [theme],
  );
}

interface StageDatum {
  name: string;
  model: string;
  totalMs: number;
  perFacingMs: number;
  share: number;
}

function useStageData(): StageDatum[] {
  return React.useMemo(
    () =>
      PIPELINE_STAGES.map((stage) => ({
        name: stage.name,
        model: stage.model,
        totalMs: stage.totalMs,
        perFacingMs: stage.perFacingMs,
        share: stage.totalMs / PIPELINE_TOTAL_MS,
      })),
    [],
  );
}

function StageTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;
  const datum = payload[0].payload as StageDatum;

  return (
    <div className="bg-popover border-border rounded-lg border p-3 text-xs shadow-lg">
      <p className="font-semibold">{datum.name}</p>
      <p className="text-muted-foreground font-mono text-2xs">{datum.model}</p>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
        <dt className="text-muted-foreground">Total</dt>
        <dd className="tabular text-right font-mono font-semibold">
          {formatDuration(datum.totalMs)}
        </dd>
        <dt className="text-muted-foreground">Per facing</dt>
        <dd className="tabular text-right font-mono font-semibold">
          {formatDuration(datum.perFacingMs)}
        </dd>
        <dt className="text-muted-foreground">Share</dt>
        <dd className="tabular text-right font-mono font-semibold">
          {formatPercent(datum.share)}
        </dd>
      </dl>
    </div>
  );
}

/**
 * Per-stage wall-clock cost.
 *
 * Horizontal bars — the stage names are long, and magnitude comparison is the
 * only job. One measure, one ordinal hue ramp, direct value labels, no legend.
 */
export function StageLatencyChart() {
  const ink = useChartInk();
  const data = useStageData();

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 76, bottom: 4, left: 4 }}>
        <XAxis type="number" hide domain={[0, "dataMax"]} />
        <YAxis
          type="category"
          dataKey="name"
          width={140}
          tickLine={false}
          axisLine={false}
          tick={{ fill: ink.muted, fontSize: 11 }}
        />
        <Tooltip
          cursor={{ fill: ink.grid, fillOpacity: 0.4 }}
          content={<StageTooltip />}
        />
        <Bar dataKey="totalMs" radius={[0, 4, 4, 0]} barSize={22} isAnimationActive={false}>
          {data.map((entry, index) => (
            <Cell key={entry.name} fill={ink.ramp[index]} />
          ))}
          {/*
            Rendered as a bare <text> rather than Recharts' built-in label:
            the default inherits the bar's width and wraps "521 ms" onto two
            lines whenever the bar is narrow.
          */}
          <LabelList
            dataKey="totalMs"
            content={({ x, y, width, height, value }) => (
              <text
                x={Number(x) + Number(width) + 8}
                y={Number(y) + Number(height) / 2}
                dominantBaseline="central"
                fill={ink.muted}
                fontSize={11}
                fontFamily="var(--font-mono)"
              >
                {formatDuration(Number(value))}
              </text>
            )}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/**
 * Share of total pipeline time.
 *
 * A 100% stacked bar rather than a donut: four ordered segments read left to
 * right against a common baseline, and close values stay comparable.
 */
export function StageShareChart() {
  const ink = useChartInk();
  const data = useStageData();

  return (
    <div className="space-y-3">
      <div className="border-border flex h-8 w-full overflow-hidden rounded-md border">
        {data.map((datum, index) => (
          <div
            key={datum.name}
            className="flex items-center justify-center"
            style={{
              width: `${datum.share * 100}%`,
              backgroundColor: ink.ramp[index],
              // 2px surface gap keeps adjacent segments visually separate.
              marginRight: index < data.length - 1 ? 2 : 0,
            }}
            title={`${datum.name}: ${formatPercent(datum.share)}`}
          >
            {datum.share > 0.12 && (
              <span className="tabular px-1 font-mono text-2xs font-bold text-white mix-blend-luminosity">
                {formatPercent(datum.share, 0)}
              </span>
            )}
          </div>
        ))}
      </div>

      <ul className="grid gap-1.5 sm:grid-cols-2">
        {data.map((datum, index) => (
          <li key={datum.name} className="flex items-center gap-2 text-xs">
            <span
              className="size-2.5 shrink-0 rounded-[3px]"
              style={{ backgroundColor: ink.ramp[index] }}
              aria-hidden
            />
            <span className="min-w-0 flex-1 truncate">{datum.name}</span>
            <span className="tabular text-muted-foreground font-mono text-2xs">
              {formatPercent(datum.share, 1)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
