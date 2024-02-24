import type { BackendModel } from "$lib/server/models";

        
interface queryOptions {
    model: BackendModel,
    question: string
}

async function query({ model, question }: queryOptions) {
	const r = await fetch(`${model.rag!.url}/query?question=${encodeURIComponent(question)}`, {
			method: "GET",
			headers: {
				"Content-Type": "application/json",
			}
        });
    return (await r.json())
 }

 export default query