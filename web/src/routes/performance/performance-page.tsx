import * as React from "react";
import { ArrowRight, Cpu, Gauge, Layers, Timer } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { PageHeader } from "@/components/common/page-header";
import { StatCard } from "@/components/common/stat-card";
import { StageLatencyChart, StageShareChart } from "./stage-charts";
import { BENCHMARK_FACINGS, PIPELINE_STAGES, PIPELINE_TOTAL_MS } from "@/config/pipeline";
import { buildAuditModel } from "@/lib/audit";
import { formatDuration, formatInteger, formatPercent } from "@/lib/format";
import { useCurrentAudit, useHealth } from "@/lib/api/queries";

/**
 * Latency and architecture reference.
 *
 * Left column: where the time goes. Right column: what each stage actually
 * does. When an audit has been run this session, its measured total is shown
 * alongside the published benchmark for comparison.
 */
export default function PerformancePage() {
  const { data: audit } = useCurrentAudit();
  const { data: health } = useHealth();

  const model = React.useMemo(() => buildAuditModel(audit), [audit]);
  const benchmarkPerFacing = PIPELINE_TOTAL_MS / BENCHMARK_FACINGS;

  return (
    <div className="mx-auto w-full max-w-[1600px] space-y-6 p-4 sm:p-6">
      <PageHeader
        title="Performance & Architecture"
        description={`Per-stage CPU cost for the shipped model configuration, measured over a ${BENCHMARK_FACINGS}-facing shelf. Use it to reason about where a slow audit is spending its time.`}
        breadcrumbs={[{ label: "Operations" }, { label: "Performance" }]}
        actions={
          health && (
            <Badge variant="outline" className="font-mono">
              gallery v{health.db_version}
            </Badge>
          )
        }
      />

      <section aria-label="Latency summary" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={Timer}
          tone="primary"
          label="Benchmark shelf total"
          value={formatDuration(PIPELINE_TOTAL_MS)}
          hint={`${BENCHMARK_FACINGS} facings, end to end`}
        />
        <StatCard
          icon={Gauge}
          tone="info"
          label="Benchmark per facing"
          value={formatDuration(benchmarkPerFacing)}
          hint="Amortised across all four stages"
        />
        <StatCard
          icon={Cpu}
          tone="warning"
          label="Embedding share"
          value={formatPercent((PIPELINE_STAGES[1].totalMs ?? 0) / PIPELINE_TOTAL_MS, 0)}
          hint="DINOv3 dominates the budget"
        />
        <StatCard
          icon={Layers}
          tone={model ? "success" : "primary"}
          label="Last measured audit"
          value={model ? formatDuration(model.processingTimeMs) : "—"}
          hint={
            model
              ? `${formatInteger(model.total)} facings · ${formatDuration(model.perFacingMs)}/facing`
              : "Run a shelf audit to compare"
          }
        />
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Wall-clock cost by stage</CardTitle>
              <CardDescription>
                Total milliseconds for one benchmark shelf. Darker means slower.
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <StageLatencyChart />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div>
              <CardTitle>Share of total time</CardTitle>
              <CardDescription>
                Feature extraction is the only stage that scales with facing count.
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <StageShareChart />
          </CardContent>
        </Card>
      </div>

      {/* Accessible table alternative to the charts above. */}
      <Card>
        <CardHeader>
          <div>
            <CardTitle>Stage breakdown</CardTitle>
            <CardDescription>
              The same figures as the charts, in tabular form.
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10">#</TableHead>
                <TableHead>Stage</TableHead>
                <TableHead>Model</TableHead>
                <TableHead className="text-right">Total</TableHead>
                <TableHead className="text-right">Per facing</TableHead>
                <TableHead className="text-right">Share</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {PIPELINE_STAGES.map((stage) => (
                <TableRow key={stage.id}>
                  <TableCell className="text-muted-foreground font-mono text-xs">
                    {stage.order}
                  </TableCell>
                  <TableCell className="font-medium">{stage.name}</TableCell>
                  <TableCell className="text-muted-foreground font-mono text-2xs">
                    {stage.model}
                  </TableCell>
                  <TableCell className="tabular text-right font-mono text-xs">
                    {formatDuration(stage.totalMs)}
                  </TableCell>
                  <TableCell className="tabular text-right font-mono text-xs">
                    {formatDuration(stage.perFacingMs)}
                  </TableCell>
                  <TableCell className="tabular text-right font-mono text-xs font-semibold">
                    {formatPercent(stage.totalMs / PIPELINE_TOTAL_MS, 1)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* --------------------------- Architecture flow ---------------------- */}
      <section aria-labelledby="architecture-heading" className="space-y-4">
        <h2 id="architecture-heading" className="text-lg font-semibold">
          Inference architecture
        </h2>
        <ol className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-4">
          {PIPELINE_STAGES.map((stage, index) => (
            <li key={stage.id} className="relative">
              <Card className="h-full">
                <CardContent className="space-y-2 p-4">
                  <div className="flex items-center gap-2">
                    <span className="bg-primary-muted text-primary flex size-6 shrink-0 items-center justify-center rounded-md font-mono text-xs font-bold">
                      {stage.order}
                    </span>
                    <h3 className="min-w-0 flex-1 truncate text-sm font-semibold">{stage.name}</h3>
                  </div>
                  <Badge variant="secondary" className="font-mono">
                    {stage.model}
                  </Badge>
                  <p className="text-muted-foreground text-xs leading-relaxed">
                    {stage.description}
                  </p>
                  <p className="tabular text-muted-foreground font-mono text-2xs">
                    {formatDuration(stage.totalMs)} · {formatDuration(stage.perFacingMs)}/facing
                  </p>
                </CardContent>
              </Card>
              {index < PIPELINE_STAGES.length - 1 && (
                <ArrowRight
                  className="text-muted-foreground absolute top-1/2 -right-3 hidden size-4 -translate-y-1/2 2xl:block"
                  aria-hidden
                />
              )}
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
