import * as React from "react";
import { Link } from "react-router-dom";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import {
  Boxes,
  CircleCheck,
  FolderOpen,
  Images,
  Lightbulb,
  Microscope,
  RotateCcw,
  Rocket,
  TriangleAlert,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input, Textarea } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PageHeader } from "@/components/common/page-header";
import { EmptyState } from "@/components/common/states";
import { CropDropzone } from "@/components/onboarding/crop-dropzone";
import { useNextClassId, useOnboardSku } from "@/lib/api/queries";
import { formatInteger, formatPercent } from "@/lib/format";
import {
  deriveDisplayName,
  ONBOARDING_DEFAULTS,
  onboardingSchema,
  PACK_TYPES,
  type OnboardingValues,
} from "./onboarding-schema";
import type { OnboardResponse } from "@/types/api";

/**
 * Pipeline 2 — few-shot SKU onboarding.
 *
 * Metadata capture and the crop picker sit in one column; the diagnostic panel
 * beside them fills in as soon as the server responds, so the operator sees the
 * embedding count, catalogue record and validation audit without navigating.
 */
export default function OnboardingPage() {
  const { data: nextClassId, isLoading: loadingClassId } = useNextClassId();
  const onboard = useOnboardSku();
  const [result, setResult] = React.useState<OnboardResponse | null>(null);

  const form = useForm<OnboardingValues>({
    resolver: zodResolver(onboardingSchema),
    defaultValues: ONBOARDING_DEFAULTS,
    mode: "onBlur",
  });

  const source = form.watch("source");
  const classId = nextClassId?.next_class_id;

  const onSubmit = form.handleSubmit((values) => {
    if (classId === undefined) {
      toast.error("Waiting for the next class id — try again in a moment.");
      return;
    }

    onboard.mutate(
      {
        class_id: classId,
        brand: values.brand,
        product_name: values.productName,
        variant: values.variant,
        size: values.size,
        pack_type: values.packType,
        display_name: values.displayName || deriveDisplayName(values),
        notes: values.notes,
        ...(values.source === "folder"
          ? { folderPath: values.folderPath }
          : { referenceImages: values.referenceImages }),
        validationShelfImage: values.validationShelfImage,
      },
      {
        onSuccess: (data) => {
          setResult(data);
          toast.success(`Class ${data.class_id ?? classId} registered`, {
            description: `${formatInteger(data.crops_added)} crops embedded into gallery v${data.version}.`,
          });
        },
        onError: (error) => {
          toast.error("Onboarding failed", {
            description: error instanceof Error ? error.message : "Unknown error",
          });
        },
      },
    );
  });

  const reset = () => {
    form.reset(ONBOARDING_DEFAULTS);
    setResult(null);
    onboard.reset();
    toast.info("Form cleared");
  };

  return (
    <div className="mx-auto w-full max-w-[1600px] space-y-6 p-4 sm:p-6">
      <PageHeader
        title="Add New SKU"
        description="Register a product from 10–50 reference crops. The service runs a quality gate, extracts DINOv3 embeddings, writes the catalogue record and — optionally — validates recognition against a live shelf photo."
        breadcrumbs={[{ label: "Catalogue", to: "/catalog" }, { label: "Add New SKU" }]}
        actions={
          <Button variant="outline" asChild>
            <Link to="/catalog">
              <Boxes aria-hidden />
              View catalogue
            </Link>
          </Button>
        }
      />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
        {/* ------------------------------- Form ------------------------------ */}
        <Form {...form}>
          <form onSubmit={onSubmit} noValidate>
            <Card>
              <CardHeader>
                <CardTitle>Commercial metadata</CardTitle>
              </CardHeader>

              <CardContent className="space-y-5">
                <div className="grid gap-4 sm:grid-cols-2">
                  <FormItem>
                    <Label htmlFor="assigned-class-id">
                      Class id
                      <Badge variant="success">auto</Badge>
                    </Label>
                    <Input
                      id="assigned-class-id"
                      readOnly
                      value={loadingClassId ? "" : (classId ?? "")}
                      placeholder="Allocating…"
                      className="text-success font-mono font-bold"
                      aria-describedby="assigned-class-id-help"
                    />
                    <FormDescription id="assigned-class-id-help">
                      Next free id across the vector store and catalogue JSONs.
                    </FormDescription>
                  </FormItem>

                  <FormField
                    control={form.control}
                    name="brand"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>
                          Brand <RequiredMark />
                        </FormLabel>
                        <FormControl>
                          <Input placeholder="e.g. Nesquik" autoComplete="off" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <FormField
                    control={form.control}
                    name="productName"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>
                          Product name <RequiredMark />
                        </FormLabel>
                        <FormControl>
                          <Input placeholder="e.g. Chocolate Drink Mix" autoComplete="off" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="variant"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Variant / flavour</FormLabel>
                        <FormControl>
                          <Input placeholder="e.g. Cocoa Powder" autoComplete="off" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <FormField
                    control={form.control}
                    name="size"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Size / weight</FormLabel>
                        <FormControl>
                          <Input placeholder="e.g. 400g" autoComplete="off" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="packType"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Pack type</FormLabel>
                        <Select value={field.value} onValueChange={field.onChange}>
                          <FormControl>
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {PACK_TYPES.map((type) => (
                              <SelectItem key={type.value} value={type.value}>
                                {type.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <FormField
                  control={form.control}
                  name="displayName"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Commercial display title</FormLabel>
                      <FormControl>
                        <Input
                          placeholder={
                            deriveDisplayName(form.getValues()) ||
                            "e.g. Nesquik Chocolate Drink Mix - 400g Pouch"
                          }
                          autoComplete="off"
                          {...field}
                        />
                      </FormControl>
                      <FormDescription>
                        Shown on bounding boxes and in the review queue. Leave blank to build it
                        from brand, product, variant and size.
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="notes"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Description &amp; notes</FormLabel>
                      <FormControl>
                        <Textarea
                          rows={2}
                          placeholder="Packaging cues, look-alike SKUs to watch for…"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Separator />

                {/* --------------------------- Reference crops -------------- */}
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h3 className="text-sm font-semibold">Reference crops</h3>
                    <FormField
                      control={form.control}
                      name="source"
                      render={({ field }) => (
                        <Tabs value={field.value} onValueChange={field.onChange}>
                          <TabsList aria-label="Reference crop source">
                            <TabsTrigger value="files">
                              <Images aria-hidden />
                              Upload files
                            </TabsTrigger>
                            <TabsTrigger value="folder">
                              <FolderOpen aria-hidden />
                              Server folder
                            </TabsTrigger>
                          </TabsList>
                        </Tabs>
                      )}
                    />
                  </div>

                  {source === "files" ? (
                    <FormField
                      control={form.control}
                      name="referenceImages"
                      render={({ field }) => (
                        <FormItem>
                          <FormControl>
                            <CropDropzone
                              files={field.value}
                              onChange={field.onChange}
                              disabled={onboard.isPending}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  ) : (
                    <FormField
                      control={form.control}
                      name="folderPath"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Server folder path</FormLabel>
                          <FormControl>
                            <Input placeholder="data/Nesquik" className="font-mono" {...field} />
                          </FormControl>
                          <FormDescription>
                            Relative to the workspace root. Every image in the folder is treated as
                            a reference crop for this class.
                          </FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  )}
                </div>

                <Separator />

                {/* ------------------------ Validation shelf ---------------- */}
                <FormField
                  control={form.control}
                  name="validationShelfImage"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Validation shelf image (optional)</FormLabel>
                      <FormControl>
                        <Input
                          type="file"
                          accept="image/*"
                          onChange={(event) => field.onChange(event.target.files?.[0] ?? null)}
                        />
                      </FormControl>
                      <FormDescription>
                        A shelf photo containing facings of this SKU. The service audits it right
                        after registration to tell you whether the crops were enough.
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </CardContent>

              <div className="border-border flex flex-col gap-2 border-t p-5 sm:flex-row">
                <Button
                  type="button"
                  variant="ghost"
                  onClick={reset}
                  disabled={onboard.isPending}
                  className="text-destructive"
                >
                  <RotateCcw aria-hidden />
                  Reset form
                </Button>
                <Button type="submit" className="flex-1" loading={onboard.isPending}>
                  <Rocket aria-hidden />
                  {onboard.isPending ? "Embedding crops…" : "Register SKU & run validation"}
                </Button>
              </div>
            </Card>
          </form>
        </Form>

        {/* ------------------------------ Diagnostics ------------------------ */}
        <div className="xl:sticky xl:top-20 xl:self-start">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Microscope className="text-primary size-4 shrink-0" aria-hidden />
                Onboarding diagnostics
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {result ? (
                <OnboardResult result={result} fallbackClassId={classId} />
              ) : (
                <EmptyState
                  icon={Microscope}
                  title="Awaiting submission"
                  description="Fill in the catalogue metadata and pick reference crops. Embedding counts, the registered catalogue card and the shelf validation benchmark appear here."
                />
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function RequiredMark() {
  return (
    <span className="text-destructive" aria-hidden>
      *
    </span>
  );
}

function OnboardResult({
  result,
  fallbackClassId,
}: {
  result: OnboardResponse;
  fallbackClassId?: number;
}) {
  const metadata = result.metadata ?? {};
  const validation = result.validation_audit;

  return (
    <div className="divide-border divide-y">
      <div className="p-5">
        <Alert variant="success">
          <CircleCheck aria-hidden />
          <AlertTitle>Registered successfully</AlertTitle>
          <AlertDescription>
            {result.message ?? "Written to the commercial catalogue and SQLite vector gallery."}
          </AlertDescription>
        </Alert>
      </div>

      <dl className="grid grid-cols-2 gap-4 p-5">
        <Stat label="Crops embedded" value={formatInteger(result.crops_added)} tone="text-success" />
        <Stat label="Gallery version" value={`v${result.version}`} tone="text-info" />
        <Stat label="Vector dimension" value="768-D" tone="text-primary" />
        <Stat
          label="Class id"
          value={String(result.class_id ?? fallbackClassId ?? "—")}
          tone="text-admin"
        />
      </dl>

      {Object.keys(metadata).length > 0 && (
        <div className="p-5">
          <h4 className="text-muted-foreground mb-2 text-2xs font-semibold tracking-wide uppercase">
            Registered catalogue card
          </h4>
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
            {Object.entries(metadata).map(([key, value]) => (
              <React.Fragment key={key}>
                <dt className="text-muted-foreground capitalize">{key.replaceAll("_", " ")}</dt>
                <dd className="truncate font-medium">{String(value) || "—"}</dd>
              </React.Fragment>
            ))}
          </dl>
        </div>
      )}

      {validation && (
        <div className="space-y-3 p-5">
          <h4 className="text-muted-foreground text-2xs font-semibold tracking-wide uppercase">
            Shelf recognition benchmark
          </h4>
          <div className="grid grid-cols-2 gap-4">
            <Stat
              label="Facings recognised"
              value={formatInteger(validation.facings_detected)}
              tone="text-success"
            />
            <Stat
              label="Mean visual similarity"
              value={formatPercent(validation.mean_similarity)}
              tone="text-info"
            />
          </div>
          <Alert variant={validation.pass_validation ? "info" : "warning"}>
            {validation.pass_validation ? (
              <Lightbulb aria-hidden />
            ) : (
              <TriangleAlert aria-hidden />
            )}
            <AlertTitle>
              {validation.pass_validation ? "Ready for automated audit" : "More coverage recommended"}
            </AlertTitle>
            <AlertDescription>{validation.recommendation}</AlertDescription>
          </Alert>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div>
      <dt className="text-muted-foreground text-2xs font-medium">{label}</dt>
      <dd className={`tabular text-xl leading-tight font-bold ${tone}`}>{value}</dd>
    </div>
  );
}
