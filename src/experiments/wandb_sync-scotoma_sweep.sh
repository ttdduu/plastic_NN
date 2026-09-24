#!/bin/bash

# Function to sync a run and mark it as synced
sync_run() {
    local run_path="$1"
    echo "Syncing run: $run_path"
    wandb sync "$run_path"
    synced_runs[$run_path]=1
}

# Initialize associative array for tracking synced runs
declare -A synced_runs

# First, sync all existing runs in the latest experiment directory
latest_exp=$(find experiments/ -maxdepth 1 -type d -name "2024*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")

if [ -n "$latest_exp" ]; then
    echo "Found latest experiment directory: $latest_exp"
    echo "Performing initial sync of all existing runs..."
    
    # Get all runs and sort them by creation time
    readarray -t existing_runs < <(find "$latest_exp/wandb/" -name "offline-*" -type d | sort -t'-' -k3)
    
    # Sync all existing runs except the last one
    for ((i=0; i<${#existing_runs[@]}-1; i++)); do
        sync_run "${existing_runs[$i]}"
    done
    
    # Set the last run as current_run if it exists
    if [ ${#existing_runs[@]} -gt 0 ]; then
        current_run="${existing_runs[-1]}"
        echo "Setting current run to: $current_run"
    fi
fi

# Continue with the original monitoring loop
while true; do
    echo "Checking for new wandb runs..."
    sleep 10
    
    # Find the most recent experiment directory
    latest_exp=$(find experiments/ -maxdepth 1 -type d -name "2024*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
    
    if [ -n "$latest_exp" ]; then
        # Get all runs sorted by their creation time
        readarray -t all_runs < <(find "$latest_exp/wandb/" -name "offline-*" -type d | sort -t'-' -k3)
        
        if [ ${#all_runs[@]} -gt 0 ]; then
            # Get the last run from the sorted array
            newest_run="${all_runs[-1]}"
            
            if [ -n "$newest_run" ] && [ "$newest_run" != "$current_run" ] && [ -z "${synced_runs[$newest_run]}" ]; then
                echo "New run detected: $newest_run"
                
                if [ -n "$current_run" ]; then
                    # Sync the previous run one final time
                    sync_run "$current_run"
                fi
                
                # Update our tracking variables
                current_run="$newest_run"
            fi
            
            # Sync current run if it exists and hasn't been marked as fully synced
            if [ -n "$current_run" ] && [ -z "${synced_runs[$current_run]}" ]; then
                echo "Syncing current run: $current_run"
                wandb sync "$current_run"
            fi
        fi
    fi
    
    # Check if training has finished
    if [ ! -f "plans.txt" ]; then
        echo "Training has finished (scotoma.err no longer exists)"
        
        # Final sync of the last run
        if [ -n "$current_run" ]; then
            sync_run "$current_run"
        fi
        
        echo "Final sync completed"
        break
    fi
done