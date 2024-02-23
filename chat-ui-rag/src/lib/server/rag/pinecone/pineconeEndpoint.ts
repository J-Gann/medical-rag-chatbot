import type { BackendModel } from "$lib/server/models";
import type { queryOptions } from "../endpoints"
        

async function pineconeQuery({ model, question }: queryOptions) {
	const r = await fetch(`${model.rag!.url}/query?question=${encodeURIComponent(question)}`, {
			method: "GET",
			headers: {
				"Content-Type": "application/json",
			}
        });
    return (await r.json()).answer
 }

 export default pineconeQuery