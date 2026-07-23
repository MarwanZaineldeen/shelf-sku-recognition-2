import { z } from "zod";
import { MAX_CROPS, MIN_CROPS } from "@/components/onboarding/crop-dropzone";

export const PACK_TYPES = [
  { value: "box", label: "Box / carton" },
  { value: "pouch/container", label: "Pouch / container" },
  { value: "bottle", label: "Bottle" },
  { value: "can", label: "Can / tin" },
  { value: "bag", label: "Bag" },
] as const;

const packTypeValues = PACK_TYPES.map((type) => type.value) as [string, ...string[]];

/**
 * Onboarding form contract.
 *
 * Reference crops can arrive either as browser files or as a server-side folder
 * path; exactly one of those must be satisfied, which is what the refinement
 * enforces.
 */
export const onboardingSchema = z
  .object({
    brand: z.string().trim().min(1, "Brand is required"),
    productName: z.string().trim().min(1, "Product name is required"),
    variant: z.string().trim(),
    size: z.string().trim(),
    packType: z.enum(packTypeValues),
    displayName: z.string().trim(),
    notes: z.string().trim(),
    source: z.enum(["files", "folder"]),
    folderPath: z.string().trim(),
    referenceImages: z.array(z.custom<File>((value) => value instanceof File)),
    validationShelfImage: z.custom<File | null>((value) => value === null || value instanceof File),
  })
  .superRefine((values, ctx) => {
    if (values.source === "folder") {
      if (!values.folderPath) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["folderPath"],
          message: "Enter a folder path readable by the server, e.g. data/Nesquik",
        });
      }
      return;
    }

    const count = values.referenceImages.length;
    if (count < MIN_CROPS || count > MAX_CROPS) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["referenceImages"],
        message: `Select between ${MIN_CROPS} and ${MAX_CROPS} reference crops (currently ${count}).`,
      });
    }
  });

export type OnboardingValues = z.infer<typeof onboardingSchema>;

export const ONBOARDING_DEFAULTS: OnboardingValues = {
  brand: "",
  productName: "",
  variant: "",
  size: "",
  packType: "box",
  displayName: "",
  notes: "",
  source: "files",
  folderPath: "",
  referenceImages: [],
  validationShelfImage: null,
};

/** Fallback commercial title when the operator leaves the field blank. */
export function deriveDisplayName(values: Pick<OnboardingValues, "brand" | "productName" | "variant" | "size">) {
  return [values.brand, values.productName, values.variant, values.size]
    .map((part) => part.trim())
    .filter(Boolean)
    .join(" ");
}
