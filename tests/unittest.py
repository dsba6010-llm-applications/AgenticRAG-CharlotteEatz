from backend.agent.tools.general import book_a_cab, book_a_table, answer_question

# # Test book_a_cab
# def test_book_a_cab():
#     result = book_a_cab("I need a taxi to go to the restaurant")
#     assert result == "your taxi has been booked", "Cab booking response is incorrect"
#     print("Test Passed: The cab booking response is correct")


def test_book_a_cab():
    result = book_a_cab("I need a taxi to go to the restaurant")
    assert result == "your taxi has been booked", "Cab booking response is incorrect"


test_book_a_cab()
# # Test book_a_table
# def test_book_a_table():
#     result = book_a_table("Please book a table for John Doe at 7 PM")
#     assert result == "Restaurant table as been booked for requetsed party", "Table booking response is incorrect"

# # Tests for book_a_cab
# @pytest.mark.parametrize("input_query, expected_response", [
#     ("I need a taxi to go to the restaurant", "your taxi has been booked"),
#     ("Taxi to the airport", "your taxi has been booked"),
#     ("", "your taxi has been booked"),  # Edge case: empty string
#     ("1234567890", "your taxi has been booked"),  # Input with only numbers
#     ("!@#$%^&*", "your taxi has been booked"),  # Special characters
# ])
# def test_book_a_cab_varied_inputs(input_query, expected_response):
#     result = book_a_cab(input_query)
#     assert result == expected_response, f"Unexpected response for query: {input_query}"

# #  Tests for book_a_table
# @pytest.mark.parametrize("input_query, expected_response", [
#     ("Please book a table for John Doe at 7 PM", "Restaurant table as been booked for requetsed party"),
#     ("Book table for Jane at 9 AM", "Restaurant table as been booked for requetsed party"),
#     ("", "Restaurant table as been booked for requetsed party"),  # Edge case: empty string
#     ("No party size mentioned", "Restaurant table as been booked for requetsed party"),
#     ("#BookTableForMe!", "Restaurant table as been booked for requetsed party"),  # Special characters
# ])
# def test_book_a_table_varied_inputs(input_query, expected_response):
#     result = book_a_table(input_query)
#     assert result == expected_response, f"Unexpected response for query: {input_query}"

# # Additional Tests for answer_question with edge cases
# @patch("backend.agent.tools.general.OpenAI")  # Mock OpenAI
# @patch("backend.agent.tools.general.OpenAIEmbeddings")  # Mock OpenAIEmbeddings
# @patch("backend.agent.tools.general.FAISS.load_local")  # Mock FAISS.load_local
# @pytest.mark.parametrize("input_query, mock_docs, expected_response", [
#     # Normal case
#     ("What is the capital of France?", [("Paris is the capital.", 0.9)], "This is the response from RAG"),
#     # Case where FAISS returns no results
#     ("Unknown query", [], "This is the response from RAG"),
#     # Edge case: Empty query string
#     ("", [("No results for empty query.", 0.1)], "This is the response from RAG"),
#     # Edge case: Query with special characters
#     ("??????", [("No context found.", 0.2)], "This is the response from RAG")
# ])
# def test_answer_question_varied_inputs(mock_faiss_load_local, mock_openai_embeddings, mock_openai, input_query, mock_docs, expected_response):
#     # Mock environment variable
#     os.environ["OPENAI_API_KEY"] = "fake-api-key"

#     # Mock FAISS database behavior
#     mock_db = MagicMock()
#     mock_faiss_load_local.return_value = mock_db

#     # Mock embedding model
#     mock_embeddings = MagicMock()
#     mock_openai_embeddings.return_value = mock_embeddings
#     mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]  # Fake embedding

#     # Mock FAISS search response based on input
#     mock_db.max_marginal_relevance_search_with_score_by_vector.return_value = [
#         (MagicMock(page_content=doc), score) for doc, score in mock_docs
#     ]

#     # Mock LLM (OpenAI) response
#     mock_llm = MagicMock()
#     mock_openai.return_value = mock_llm
#     mock_llm.invoke.return_value = "This is the response from RAG"

#     # Call the function
#     result = answer_question(input_query)

#     # Assertions
#     assert result == expected_response, f"Unexpected response for query: {input_query}"
#     mock_faiss_load_local.assert_called_once_with(
#         os.path.dirname(os.path.abspath(__file__)) + "/../faiss_store/",
#         mock_embeddings,
#         allow_dangerous_deserialization=True
#     )
#     mock_embeddings.embed_query.assert_called_once_with(input_query)
#     mock_db.max_marginal_relevance_search_with_score_by_vector.assert_called_once()
#     mock_llm.invoke.assert_called_once()

# # Test FAISS store load failure for empty query
# @patch("backend.agent.tools.general.OpenAIEmbeddings")  # Mock OpenAIEmbeddings
# @patch("backend.agent.rag.FAISS.load_local")  # Mock FAISS.load_local
# def test_answer_question_empty_query_faiss_load_error(mock_faiss_load_local, mock_openai_embeddings):
#     # Mock environment variable
#     os.environ["OPENAI_API_KEY"] = "fake-api-key"

#     # Set up FAISS to raise an exception when loading
#     mock_faiss_load_local.side_effect = Exception("FAISS load error")

#     # Call the function and assert an exception is raised for empty query
#     with pytest.raises(Exception, match="Error loading FAISS store: FAISS load error"):
#         answer_question("")

# # Test FAISS store load with no documents
# @patch("backend.agent.tools.general.OpenAI")  # Mock OpenAI
# @patch("backend.agent.tools.general.OpenAIEmbeddings")  # Mock OpenAIEmbeddings
# @patch("backend.agent.tools.general.FAISS.load_local")  # Mock FAISS.load_local
# def test_answer_question_no_documents(mock_faiss_load_local, mock_openai_embeddings, mock_openai):
#     # Mock environment variable
#     os.environ["OPENAI_API_KEY"] = "fake-api-key"

#     # Mock FAISS database behavior with no documents
#     mock_db = MagicMock()
#     mock_faiss_load_local.return_value = mock_db

#     # Mock embedding model
#     mock_embeddings = MagicMock()
#     mock_openai_embeddings.return_value = mock_embeddings
#     mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]  # Fake embedding

#     # Mock FAISS search response with no documents
#     mock_db.max_marginal_relevance_search_with_score_by_vector.return_value = []

#     # Mock LLM (OpenAI) response
#     mock_llm = MagicMock()
#     mock_openai.return_value = mock_llm
#     mock_llm.invoke.return_value = "No results found for this query"

#     # Call the function
#     result = answer_question("Where is the moon?")

#     # Assertions
#     assert result == "No results found for this query", "Unexpected response when no documents are found"
#     mock_faiss_load_local.assert_called_once_with(
#         os.path.dirname(os.path.abspath(__file__)) + "/../faiss_store/",
#         mock_embeddings,
#         allow_dangerous_deserialization=True
#     )
#     mock_db.max_marginal_relevance_search_with_score_by_vector.assert_called_once()
#     mock_llm.invoke.assert_called_once()
