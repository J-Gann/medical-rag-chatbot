import pineconeQuery from "./pinecone/pineconeEndpoint"
import type { BackendModel } from "$lib/server/models";

export interface queryOptions {
    model: BackendModel,
    question: string
}

export async function query({ model, question }: queryOptions) {
    switch (model.rag!.vectorStoreType) {
        case "pinecone": {
            return pineconeQuery({ model, question })
        }
    }

}

export default query