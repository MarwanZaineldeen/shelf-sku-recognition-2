import { Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { EmptyState } from "@/components/common/states";
import { CropThumb } from "./crop-thumb";
import { exemplarUrl } from "@/lib/api/endpoints";
import { formatScore } from "@/lib/format";
import { UNKNOWN_CLASS_ID, type Candidate } from "@/types/api";
import { ListOrdered } from "lucide-react";

/**
 * Class-unique retrieval slate. The row the VLM reranker promoted is called
 * out explicitly so a reviewer can see when text evidence overrode raw visual
 * similarity.
 */
export function CandidateTable({ candidates }: { candidates: Candidate[] }) {
  if (candidates.length === 0) {
    return (
      <EmptyState
        size="sm"
        icon={ListOrdered}
        title="No candidate slate"
        description="This facing was resolved without a retrieval shortlist."
      />
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-10">#</TableHead>
          <TableHead className="w-14">Exemplar</TableHead>
          <TableHead>Commercial SKU</TableHead>
          <TableHead className="w-24 text-right">Visual</TableHead>
          <TableHead className="w-24 text-right">Fused</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {candidates.map((candidate, index) => {
          const isVlmPick = Boolean(candidate.vlm_selected);
          const classLabel =
            candidate.class_id === UNKNOWN_CLASS_ID ? "Unknown" : `Class ${candidate.class_id}`;

          return (
            <TableRow
              key={`${candidate.class_id}-${index}`}
              className={cn(isVlmPick && "bg-warning-subtle/60 border-l-warning border-l-2")}
            >
              <TableCell className="text-muted-foreground font-mono text-xs">{index + 1}</TableCell>
              <TableCell>
                <CropThumb
                  src={candidate.exemplar_url ?? exemplarUrl(candidate.class_id)}
                  alt={`Reference exemplar for ${candidate.display_name}`}
                  className="size-9"
                />
              </TableCell>
              <TableCell className="min-w-0">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="truncate font-medium">{candidate.display_name}</span>
                  {isVlmPick && (
                    <Badge variant="warning">
                      <Sparkles aria-hidden />
                      VLM pick
                    </Badge>
                  )}
                </div>
                <p className="text-muted-foreground font-mono text-2xs">{classLabel}</p>
              </TableCell>
              <TableCell className="tabular text-right font-mono text-xs">
                {formatScore(candidate.similarity)}
              </TableCell>
              <TableCell className="tabular text-right font-mono text-xs font-semibold">
                {formatScore(candidate.s_fused ?? candidate.similarity)}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
