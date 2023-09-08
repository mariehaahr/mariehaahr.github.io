import path from 'path'
import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
	// Type-check frontmatter using a schema
	schema: z.object({
		title: z.string(),
		description: z.string(),
		// Transform string to Date object
		pubDate: z.coerce.date(),
		updatedDate: z.coerce.date().optional(),
		heroImage: z.string().optional(),
	}),
});

const glob = import.meta.glob('./**'); /* vite */

export const collectionNames = Object.keys(glob).map((filepath) => path.basename(path.dirname(filepath)));

function assignCollection(acc: Record<string, any>, name: string): Record<string, any> {
	return Object.assign(acc, { [name]: defineCollection({ ...blog }) });
} 

export const collections = collectionNames.reduce(assignCollection, {});
