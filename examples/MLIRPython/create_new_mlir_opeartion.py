from mlir.ir import (Context,
                     Location,
                     Operation,
                     StringAttr)

# Create a new MLIR context.
context = Context()

# Create a new operation using the Python bindings.
def create_print_message_op():
    with Location.unknown(context):
        # Define the operation using the dialect assembly syntax.
        # Add any desired attributes to the operation.
        # For example, to add a string message attribute:
        message_attr = StringAttr.get("Hello, MLIR!")
        operation = Operation.create("std.print_message",
                                     operands=[],
                                     attributes={"message": message_attr})

        # Return the created operation.
        return operation


# Create the print_message operation.
print_message_op = create_print_message_op()

# Print the MLIR representation of the operation.
print(print_message_op)
